import re
from typing import List, Dict, Optional, Any
from titans_maas.ext.maas.maas.provider.base_llm import BaseLLM
from titans_maas.ext.maas.maas.logs import logger
from experiments.datasets.Combined.train.template.op_prompt import (
    COT_PROMPT, DEBATE_PROMPT, DEBATE_SYNTHESIS_PROMPT,
    SELFCONSISTENCY_PROMPT, SELFCONSISTENCY_VOTE_PROMPT,
    SELFREFINE_PROMPT, ENSEMBLE_MEMBER_PROMPT, ENSEMBLE_RANK_PROMPT,
    TESTING_PROMPT, REACT_PROMPT, EARLYEXIT_PROMPT,
)
import os

class BaseOperator:
    def __init__(self, llm: BaseLLM, prompt: Optional[str] = None):
        self.llm = llm
        self.specialized_instructions = prompt or ""

    def _extract_tool_property(self, text: str, property_name: str) -> str:
        pattern = rf"<{property_name}>(.*?)</{property_name}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        pattern_malformed = rf"<{property_name}>(.*?)(?:<|$)"
        match_malformed = re.search(pattern_malformed, text, re.DOTALL | re.IGNORECASE)
        return match_malformed.group(1).strip() if match_malformed else ""

    def _format_history_for_llm(self, history: List[Dict]) -> List[Dict]:
        formatted: List[Dict] = []
        for m in history or []:
            role = m.get("role", "user")
            if role == "assistant":
                role = "model"
            parts = m.get("parts")
            if parts is None:
                content = m.get("content", "")
                parts = [content]
            else:
                norm_parts = []
                for p in parts:
                    if isinstance(p, dict) and "text" in p:
                        norm_parts.append(p["text"])
                    else:
                        norm_parts.append(p)
                parts = norm_parts
            formatted.append({"role": role, "parts": parts})
        return formatted


class CoT(BaseOperator):
    def __init__(self, llm: BaseLLM, prompt: Optional[str] = None):
        super().__init__(llm, prompt=prompt)
        self.base_prompt = COT_PROMPT

    async def __call__(self, problem: str, context: Optional[Dict[str, Any]] = None, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []
        ctx = context or {}
        ctx_str = f"File Name: {ctx.get('file_name','')}\nFile Content: {ctx.get('file_content','')}\nFile Tree: {ctx.get('file_tree','')}"
        prompt = self.base_prompt.format(problem=problem, context=ctx_str)
        if self.specialized_instructions:
            prompt += f"\n\nSpecialized Instructions:\n{self.specialized_instructions}\n"
        chat_history.append({"role": "user", "content": prompt, "parts": [{"text": prompt}]})
        llm_history = self._format_history_for_llm(chat_history)
        # Avoid double-sending the prompt; use the formatted history only
        response = await self.llm.aask([], format_msgs=llm_history)
        usage = self.llm.cost_manager.get_costs()
        logger.info(f"[COST][CoT] cost=${usage.total_cost:.6f} tokens(prompt={usage.total_prompt_tokens}, completion={usage.total_completion_tokens})")
        chat_history.append({"role": "assistant", "content": response, "parts": [{"text": response}]})
        return {"response": response, "tool_results": {}, "error": None, "chat_history": chat_history}


class Debate(BaseOperator):
    def __init__(self, llm: BaseLLM, prompt: Optional[str] = None):
        super().__init__(llm, prompt=prompt)

    async def __call__(self, problem: str, context: Optional[Dict[str, Any]] = None, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []
        transcript_parts: List[str] = []
        for rnd in range(1, 3):
            for deb in range(3):
                p = DEBATE_PROMPT.format(debater_id=deb, problem=problem, round_num=rnd)
                if self.specialized_instructions:
                    p += f"\n\nSpecialized Instructions:\n{self.specialized_instructions}\n"
                chat_history.append({"role": "user", "content": p, "parts": [{"text": p}]})
                llm_history = self._format_history_for_llm(chat_history)
                # Use only formatted chat history to avoid double-counting tokens
                stmt = await self.llm.aask([], format_msgs=llm_history)
                usage = self.llm.cost_manager.get_costs()
                logger.info(f"[COST][Debate] cost=${usage.total_cost:.6f} tokens(prompt={usage.total_prompt_tokens}, completion={usage.total_completion_tokens})")
                chat_history.append({"role": "assistant", "content": stmt, "parts": [{"text": stmt}]})
                transcript_parts.append(f"[R{rnd} D{deb}] {stmt}")
        transcript = "\n".join(transcript_parts)
        synth = DEBATE_SYNTHESIS_PROMPT.format(transcript=transcript)
        chat_history.append({"role": "user", "content": synth, "parts": [{"text": synth}]})
        llm_history = self._format_history_for_llm(chat_history)
        resp = await self.llm.aask([], format_msgs=llm_history)
        usage = self.llm.cost_manager.get_costs()
        logger.info(f"[COST][DebateSynth] cost=${usage.total_cost:.6f} tokens(prompt={usage.total_prompt_tokens}, completion={usage.total_completion_tokens})")
        chat_history.append({"role": "assistant", "content": resp, "parts": [{"text": resp}]})
        return {"response": resp, "tool_results": {}, "error": None, "chat_history": chat_history}


class SelfConsistency(BaseOperator):
    def __init__(self, llm: BaseLLM, prompt: Optional[str] = None):
        super().__init__(llm, prompt=prompt)

    async def __call__(self, problem: str, context: Optional[Dict[str, Any]] = None, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []
        ctx = context or {}
        ctx_str = f"File Name: {ctx.get('file_name','')}\nFile Content: {ctx.get('file_content','')}\nFile Tree: {ctx.get('file_tree','')}"
        answers: List[str] = []
        for _ in range(5):
            p = COT_PROMPT.format(problem=problem, context=ctx_str)
            if self.specialized_instructions:
                p += f"\n\nSpecialized Instructions:\n{self.specialized_instructions}\n"
            chat_history.append({"role": "user", "content": p, "parts": [{"text": p}]})
            llm_history = self._format_history_for_llm(chat_history)
            r = await self.llm.aask([], format_msgs=llm_history)
            usage = self.llm.cost_manager.get_costs()
            logger.info(f"[COST][SelfConsistency] cost=${usage.total_cost:.6f} tokens(prompt={usage.total_prompt_tokens}, completion={usage.total_completion_tokens})")
            chat_history.append({"role": "assistant", "content": r, "parts": [{"text": r}]})
            answers.append(r)
        candidates = "\n\n".join([f"Candidate {i+1}:\n{a}" for i, a in enumerate(answers)])
        vote = SELFCONSISTENCY_VOTE_PROMPT.format(candidates=candidates)
        chat_history.append({"role": "user", "content": vote, "parts": [{"text": vote}]})
        llm_history = self._format_history_for_llm(chat_history)
        resp = await self.llm.aask([], format_msgs=llm_history)
        usage = self.llm.cost_manager.get_costs()
        logger.info(f"[COST][SelfConsistencyVote] cost=${usage.total_cost:.6f} tokens(prompt={usage.total_prompt_tokens}, completion={usage.total_completion_tokens})")
        chat_history.append({"role": "assistant", "content": resp, "parts": [{"text": resp}]})
        return {"response": resp, "tool_results": {}, "error": None, "chat_history": chat_history}


class SelfRefine(BaseOperator):
    def __init__(self, llm: BaseLLM, prompt: Optional[str] = None):
        super().__init__(llm, prompt=prompt)
        self.max_iter = 5

    async def __call__(self, problem: str, context: Optional[Dict[str, Any]] = None, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []
        ctx = context or {}
        ctx_str = f"File Name: {ctx.get('file_name','')}\nFile Content: {ctx.get('file_content','')}\nFile Tree: {ctx.get('file_tree','')}"
        p0 = COT_PROMPT.format(problem=problem, context=ctx_str)
        if self.specialized_instructions:
            p0 += f"\n\nSpecialized Instructions:\n{self.specialized_instructions}\n"
        chat_history.append({"role": "user", "content": p0, "parts": [{"text": p0}]})
        llm_history = self._format_history_for_llm(chat_history)
        sol = await self.llm.aask([], format_msgs=llm_history)
        usage = self.llm.cost_manager.get_costs()
        logger.info(f"[COST][SelfRefine:init] cost=${usage.total_cost:.6f} tokens(prompt={usage.total_prompt_tokens}, completion={usage.total_completion_tokens})")
        chat_history.append({"role": "assistant", "content": sol, "parts": [{"text": sol}]})
        for _ in range(self.max_iter):
            p = SELFREFINE_PROMPT.format(problem=problem, solution=sol)
            chat_history.append({"role": "user", "content": p, "parts": [{"text": p}]})
            llm_history = self._format_history_for_llm(chat_history)
            sol = await self.llm.aask([], format_msgs=llm_history)
            usage = self.llm.cost_manager.get_costs()
            logger.info(f"[COST][SelfRefine:iter] cost=${usage.total_cost:.6f} tokens(prompt={usage.total_prompt_tokens}, completion={usage.total_completion_tokens})")
            chat_history.append({"role": "assistant", "content": sol, "parts": [{"text": sol}]})
        return {"response": sol, "tool_results": {}, "error": None, "chat_history": chat_history}


class Ensemble(BaseOperator):
    def __init__(self, llm: BaseLLM, prompt: Optional[str] = None):
        super().__init__(llm, prompt=prompt)

    async def __call__(self, problem: str, context: Optional[Dict[str, Any]] = None, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []
        answers: List[str] = []
        for src in ["A", "B", "C"]:
            p = ENSEMBLE_MEMBER_PROMPT.format(source=src, problem=problem)
            if self.specialized_instructions:
                p += f"\n\nSpecialized Instructions:\n{self.specialized_instructions}\n"
            chat_history.append({"role": "user", "content": p, "parts": [{"text": p}]})
            llm_history = self._format_history_for_llm(chat_history)
            a = await self.llm.aask([], format_msgs=llm_history)
            usage = self.llm.cost_manager.get_costs()
            logger.info(f"[COST][Ensemble:member] cost=${usage.total_cost:.6f} tokens(prompt={usage.total_prompt_tokens}, completion={usage.total_completion_tokens})")
            chat_history.append({"role": "assistant", "content": a, "parts": [{"text": a}]})
            answers.append(a)
        joined = "\n\n".join([f"Answer {i+1}:\n{a}" for i, a in enumerate(answers)])
        rank = ENSEMBLE_RANK_PROMPT.format(answers=joined)
        chat_history.append({"role": "user", "content": rank, "parts": [{"text": rank}]})
        llm_history = self._format_history_for_llm(chat_history)
        resp = await self.llm.aask([], format_msgs=llm_history)
        usage = self.llm.cost_manager.get_costs()
        logger.info(f"[COST][Ensemble:rank] cost=${usage.total_cost:.6f} tokens(prompt={usage.total_prompt_tokens}, completion={usage.total_completion_tokens})")
        chat_history.append({"role": "assistant", "content": resp, "parts": [{"text": resp}]})
        return {"response": resp, "tool_results": {}, "error": None, "chat_history": chat_history}


class Testing(BaseOperator):
    def __init__(self, llm: BaseLLM, prompt: Optional[str] = None):
        super().__init__(llm, prompt=prompt)
        self.base_prompt = TESTING_PROMPT

    async def __call__(self, problem: str, solution: str, context: Optional[Dict[str, Any]] = None, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []
        p = self.base_prompt.format(problem=problem, solution=solution)
        if self.specialized_instructions:
            p += f"\n\nSpecialized Instructions:\n{self.specialized_instructions}\n"
        chat_history.append({"role": "user", "content": p, "parts": [{"text": p}]})
        llm_history = self._format_history_for_llm(chat_history)
        resp = await self.llm.aask([], format_msgs=llm_history)
        usage = self.llm.cost_manager.get_costs()
        logger.info(f"[COST][Testing] cost=${usage.total_cost:.6f} tokens(prompt={usage.total_prompt_tokens}, completion={usage.total_completion_tokens})")
        chat_history.append({"role": "assistant", "content": resp, "parts": [{"text": resp}]})
        return {"response": resp, "tool_results": {}, "error": None, "chat_history": chat_history}


class ReAct(BaseOperator):
    def __init__(self, llm: BaseLLM, tools: Optional[Dict[str, Any]] = None, prompt: Optional[str] = None):
        super().__init__(llm, prompt=prompt)
        self.tools = tools or {}
        self.base_prompt = REACT_PROMPT

    async def __call__(self, problem: str, context: Optional[Dict[str, Any]] = None, chat_history: Optional[List[Dict]] = None, steps: int = 15) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []
        ctx = context or {}
        if "tool_results" not in ctx:
            ctx["tool_results"] = ""
        if "last_url" not in ctx:
            ctx["last_url"] = ""
        traj: List[str] = []
        final_answer = ""
        for _ in range(steps):
            ctx_str = (
                f"File Name: {ctx.get('file_name','')}\n"
                f"File Content: {ctx.get('file_content','')}\n"
                f"File Tree: {ctx.get('file_tree','')}\n"
                f"Tool Results: {ctx.get('tool_results','')}\n"
                f"Last URL: {ctx.get('last_url','')}"
            )
            p = self.base_prompt.format(problem=problem, context=ctx_str)
            if self.specialized_instructions:
                p += f"\n\nSpecialized Instructions:\n{self.specialized_instructions}\n"
            chat_history.append({"role": "user", "content": p, "parts": [{"text": p}]})
            llm_history = self._format_history_for_llm(chat_history)
            step = await self.llm.aask([], format_msgs=llm_history)
            usage = self.llm.cost_manager.get_costs()
            logger.info(f"[COST][ReAct] cost=${usage.total_cost:.6f} tokens(prompt={usage.total_prompt_tokens}, completion={usage.total_completion_tokens})")
            chat_history.append({"role": "assistant", "content": step, "parts": [{"text": step}]})
            thought = BaseOperator._extract_tool_property(self, step, "thought")
            action = BaseOperator._extract_tool_property(self, step, "action").lower()
            inp = BaseOperator._extract_tool_property(self, step, "input")
            obs = ""
            if action == "web":
                obs = await self._do_web(inp)
            elif action == "file":
                obs = await self._do_file(inp, ctx)
            elif action == "math":
                obs = await self._do_math(inp)
            elif action == "execute":
                obs = await self._do_execute(inp)
            elif action == "browser":
                obs = await self._do_browser(inp, ctx)
            elif action == "finish":
                final_answer = inp or final_answer
                break
            else:
                obs = ""
            traj.append(f"<step>\n<thought>{thought}</thought>\n<action>{action}</action>\n<input>{inp}</input>\n<observation>{obs}</observation>\n</step>")
            if obs:
                obs_msg = f"<observation>{obs}</observation>"
                chat_history.append({"role": "user", "content": obs_msg, "parts": [{"text": obs_msg}]})
            if obs:
                try:
                    prev = ctx.get("tool_results", "")
                    add = f"{prev}\n{obs}" if prev else obs
                    ctx["tool_results"] = add
                    # Update last_url if observation contains a URL
                    import re as _re
                    m = _re.search(r"https?://[^\s<>\"]+", obs)
                    if m:
                        ctx["last_url"] = m.group(0)
                except Exception:
                    pass
        out = "\n".join(traj)
        if not final_answer:
            final_answer = out[-4000:]
        return {"response": final_answer, "tool_results": {"trajectory": out}, "error": None, "chat_history": chat_history}

    async def _do_web(self, q: str) -> str:
        browser = self.tools.get("browser_use")
        scrape = self.tools.get("scrape_web_playwright")
        try:
            txt = (q or "").strip()
            if scrape is None and browser is None:
                return "no web tools available"
            if txt.startswith("http") and scrape:
                page = await scrape.navigate(txt)
                return (page or {}).get("content") or (page or {}).get("html") or str(page)
            import re as _re
            import urllib.parse as _uq
            m = _re.search(r'^search\s+"([^"]+)"', txt, flags=_re.IGNORECASE)
            query = None
            if m:
                query = m.group(1).strip()
            elif txt.lower().startswith("search "):
                query = txt[7:].strip()
            elif txt.lower().startswith("scrape ") and 'http' in txt:
                url_m = _re.search(r'(https?://\S+)', txt)
                if url_m and scrape:
                    page = await scrape.navigate(url_m.group(1))
                    return (page or {}).get("content") or (page or {}).get("html") or str(page)
            if query and scrape:
                engine_url = "https://www.google.com/search?q=" if (os.getenv("SERPAPI_KEY") or os.getenv("SERPAPI_API_KEY")) else "https://duckduckgo.com/?q="
                url = engine_url + _uq.quote_plus(query)
                res = await scrape.search(url=url, query=query)
                return (res or {}).get("content") or (res or {}).get("html") or str(res)
            if scrape and not query and txt:
                engine_url = "https://duckduckgo.com/?q="
                url = engine_url + _uq.quote_plus(txt)
                res = await scrape.search(url=url, query=txt)
                return (res or {}).get("content") or (res or {}).get("html") or str(res)
            if browser:
                task = f"Find and extract information for: {txt}"
                res = await browser.mcp_browser_use(task=task, max_steps=20, extract_format="markdown")
                return (res or {}).get("message") if isinstance(res, dict) else str(res)
        except Exception as e:
            return f"web error: {e}"
        return ""

    async def _do_browser(self, task: str, ctx: Dict[str, Any]) -> str:
        browser = self.tools.get("browser_use")
        try:
            if not browser:
                return "no browser_use"
            t = (task or "").strip()
            if not t:
                t = "Open target page and extract the required information."
            lower = t.lower()
            if (lower.startswith("scroll") or lower.startswith("click") or lower.startswith("fill ") or lower.startswith("type ")) and not t.startswith("http"):
                last_url = (ctx or {}).get("last_url") or ""
                if last_url:
                    t = f"Open {last_url} and then {t}. If already on the page, proceed. Extract the relevant content afterward."
            res = await browser.mcp_browser_use(task=t, max_steps=30, extract_format="markdown")
            return (res or {}).get("message") if isinstance(res, dict) else str(res)
        except Exception as e:
            return f"browser error: {e}"

    async def _do_file(self, path: str, ctx: Dict[str, Any]) -> str:
        fio = self.tools.get("file_io")
        excel = self.tools.get("excel_reader")
        pdf = self.tools.get("pdf_reader")
        try:
            if not fio:
                return "no file_io"
            in_path = (path or "").strip()
            if not in_path:
                in_path = ctx.get("file_path") or ctx.get("file_name") or ""
            if in_path and not os.path.isabs(in_path):
                base_dir = None
                split = (ctx or {}).get("gaia_split")
                if split in ("test", "validation"):
                    base_dir = os.path.abspath("gaia_data/2023/test" if split == "test" else "gaia_data/2023/validation")
                if not base_dir:
                    base_dir = os.getcwd()
                in_path = os.path.abspath(os.path.join(base_dir, in_path))
            res = await fio.read_file(in_path, context=ctx)
            if not res.get("success") and str(path).lower().endswith(".xlsx") and excel:
                res = await excel.read_excel(in_path, context=ctx)
            if not res.get("success") and str(path).lower().endswith(".pdf") and pdf:
                res = await pdf.read_pdf(in_path, context=ctx)
            if res.get("success"):
                return str(res.get("content") or res)
            return str(res)
        except Exception as e:
            return f"file error: {e}"

    async def _do_math(self, expr: str) -> str:
        calc = self.tools.get("math_calculator")
        try:
            if not calc:
                return "no math_calculator"
            r = await calc.calculate(expr)
            return str(r)
        except Exception as e:
            return f"math error: {e}"

    async def _do_execute(self, code: str) -> str:
        exe = self.tools.get("code_executor")
        try:
            if not exe:
                return "no code_executor"
            r = await exe.execute(code)
            return str(r)
        except Exception as e:
            return f"exec error: {e}"


class EarlyExit(BaseOperator):
    def __init__(self, llm: BaseLLM, prompt: Optional[str] = None):
        super().__init__(llm, prompt=prompt)
        self.base_prompt = EARLYEXIT_PROMPT

    async def __call__(self, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []
        resp = "Early exit"
        return {"response": resp, "tool_results": {}, "error": None, "chat_history": chat_history}