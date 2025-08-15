# Core prompts for new operator space

COT_PROMPT = """You are a careful reasoner. Think step by step and derive a concise final answer.

Problem:
{problem}

Context:
{context}

Strict answer policy:
 
 - Output EXACTLY the answer string with no prefixes/suffixes.
 - Do not include quotes, year, citations, or extra words.
 - For titles, return the title text only. For numbers, return only the number.
 - If the question is multiple-choice with options labeled A/B/C/D, END your overall output with exactly one line: The final answer is: X (where X is A, B, C, or D).
 - If the problem is mathematics, also append a final line with the canonical math format: $$\\boxed{{<final answer only>}}$$.
 - If the problem is a coding task (HumanEval-style), output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):

<response>
<reasoning><![CDATA[
Your step-by-step reasoning here.
]]></reasoning>
<final_answer><![CDATA[
Your final answer here.
]]></final_answer>
</response>
"""

DEBATE_PROMPT = """You are Debater {debater_id} in a 3-debater, 2-round debate. Argue towards the correct answer.

Problem:
{problem}

Round {round_num} Statement:
<response>
<statement><![CDATA[
Your short statement.
]]></statement>
</response>
"""

DEBATE_SYNTHESIS_PROMPT = """
Given the following debate transcript, synthesize the key arguments presented by each side and identify the core points of contention. Based on this synthesis, provide a concise and definitive answer to the central question debated. The Combined dataset often involves nuanced arguments and requires careful consideration of context to avoid misrepresentation. Pay close attention to identifying the underlying assumptions and logical fallacies employed by each side.

Transcript:
{transcript}

Strict answer policy:
 
 - Output EXACTLY the answer string with no prefixes/suffixes.
 - Do not include quotes, year, citations, or extra words.
 - For titles, return the title text only. For numbers, return only the number.
 - If the question is multiple-choice with options labeled A/B/C/D, END your overall output with exactly one line: The final answer is: X.
 - If the problem is mathematics, also append a final line: $$oxed{{<final answer only>}}$$.
 - If the problem is a coding task, output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):

<response>
<final_answer><![CDATA[
The answer is ...
]]></final_answer>
</response>
"""

SELFCONSISTENCY_PROMPT = """Generate five independent CoT answers.

Problem:
{problem}

Context:
{context}

Return five <answer> blocks.
Each <answer> must be ONLY the exact answer string (no extra text, no quotes or year).
If the question is multiple-choice (A/B/C/D), the final chosen answer should be provided later in the vote step as: The final answer is: X.
If the problem is mathematics, ensure each answer is suitable to be placed inside $$\\boxed{{...}}$$ without extra words.
"""

SELFCONSISTENCY_VOTE_PROMPT = """
Given the diverse nature of the Combined dataset, which includes tasks ranging from commonsense reasoning and reading comprehension to mathematical problem-solving and code generation, we need a robust self-consistency voting prompt that can effectively handle the varying output formats and evaluation metrics. The key challenges are: (1) diverse output formats (text, numerical answers, code snippets), (2) varying levels of reasoning complexity, and (3) the need to consistently identify the most reliable answer across different candidate solutions.

To address these challenges, we refine the self-consistency voting prompt to explicitly consider the type of question and the expected output format. We also add instructions to prioritize answers that demonstrate coherent reasoning and consistency with known facts or mathematical principles. For mathematical problems, we emphasize the importance of verifying the correctness of the final numerical answer. For code generation tasks, we prioritize solutions that compile and produce the correct output for a given set of test cases.

Vote the most consistent final answer among the following candidates, considering the type of question and the expected output format. Prioritize answers that demonstrate coherent reasoning and consistency with known facts or mathematical principles. For mathematical problems, verify the correctness of the final numerical answer. For code generation tasks, prioritize solutions that compile and produce the correct output.

{candidates}

<response>
<voted><![CDATA[
Your chosen answer.
]]></voted>
</response>

Additionally, if the question is multiple-choice with options A/B/C/D, END your overall output with exactly one line: The final answer is: X.
If the problem is mathematics, also append a final line: $$oxed{{<final answer only>}}$$.
"""

SELFREFINE_PROMPT = """You will iteratively refine a solution up to five iterations.

Problem:
{problem}

Current Solution:
{solution}

Strict answer policy:
 - Final refined answer must be EXACT only the answer string.
 - If the question is multiple-choice A/B/C/D, END your overall output with exactly one line: The final answer is: X.
 - If the problem is mathematics, also append a final line: $$\\boxed{{<final answer only>}}$$.
 - If the problem is a coding task, output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):

<response>
<notes><![CDATA[
Your improvement notes for this iteration.
]]></notes>
<refined><![CDATA[
Refined answer.
]]></refined>
</response>
"""

ENSEMBLE_MEMBER_PROMPT = """Act as agent {source}. Answer succinctly.

Problem:
{problem}

Strict answer policy:
 - Provide ONLY the exact final answer string in <answer>.
 - No quotes, no year, no extra words.
 - If the question is multiple-choice with options A/B/C/D, END your overall output with exactly one line: The final answer is: X.
 - If the problem is mathematics, also append a final line: $$\\boxed{{<final answer only>}}$$.
 - If the problem is a coding task, output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):

<response>
<answer><![CDATA[
Your answer.
]]></answer>
</response>
"""

ENSEMBLE_RANK_PROMPT = """
Given the following set of answers generated for a single question from the Combined dataset, perform a pairwise ranking to determine the most accurate, relevant, and helpful response. The Combined dataset is characterized by its diverse range of question types, spanning factual queries, reasoning-based problems, and creative writing prompts. This heterogeneity presents a significant challenge, as a single ranking strategy may not be optimal for all question types. Furthermore, the dataset contains answers of varying lengths and styles, some exhibiting biases or factual inaccuracies.

Therefore, when ranking, consider the following criteria:

*   **Accuracy:** Prioritize answers that are factually correct and align with established knowledge. Verify information against reliable sources if necessary.
*   **Relevance:** Assess how well each answer directly addresses the question's core intent. Discard answers that are tangential or irrelevant.
*   **Completeness:** Favor answers that provide a comprehensive and thorough response, addressing all aspects of the question.
*   **Clarity and Coherence:** Rank higher answers that are well-written, easy to understand, and logically organized.
*   **Helpfulness:** Consider the overall utility of the answer in satisfying the user's information need.

Specifically, for each pair of answers, evaluate which answer is superior based on the above criteria. Use this pairwise comparison to construct a ranked list of answers. Finally, select the single best answer as the winner. Justify your choice in a brief sentence.

{answers}

<response>
<final><![CDATA[
Winner answer.
]]></final>
</response>
\nAdditionally, adhere to final formatting:
- For multiple-choice (A/B/C/D), END with: The final answer is: X.
- For mathematics, also append: $$\\boxed{{<final answer only>}}$$.
"""

TESTING_PROMPT = """
Given the following problem and proposed solution, generate a comprehensive and minimal test plan focusing on robustness, edge cases, and potential failure points. The Combined dataset often involves complex reasoning, multi-step problem-solving, and the integration of diverse knowledge domains. Therefore, your test plan should prioritize evaluating the solution's accuracy, completeness, and ability to handle ambiguous or incomplete information. Consider testing for common errors such as logical fallacies, factual inaccuracies, and inconsistencies in reasoning. Also, evaluate the solution's ability to generalize to unseen variations of the problem. The test plan should include specific test cases with expected outputs, focusing on scenarios that would expose weaknesses in the solution's underlying reasoning process. Prioritize tests that validate the core assumptions and critical steps of the solution.

Problem:
{problem}

Solution:
{solution}

<response>
<tests><![CDATA[
- Test 1: ...
- Test 2: ...
]]></tests>
</response>
"""

REACT_PROMPT = """
You are a ReAct agent specifically tuned for the Combined dataset, which includes a diverse range of tasks requiring different reasoning and tool usage strategies. This dataset is characterized by its heterogeneity, encompassing tasks that demand web search, interactive browsing, file manipulation, mathematical calculations, code execution, and a combination of these. The challenges lie in accurately identifying the appropriate tool for each task, effectively utilizing the tool, and synthesizing information from multiple sources to arrive at the correct final answer. Pay close attention to the problem description and context to determine the optimal sequence of actions. This dataset often presents ambiguous or underspecified problems, requiring you to actively clarify assumptions and seek missing information. Be prepared to handle noisy or incomplete data and to adapt your strategy based on intermediate results. Prioritize robustness and error handling in your tool usage.

In each step:
- Think briefly and strategically about the best course of action. Consider the problem's complexity and the available tools. Explicitly consider potential edge cases and failure modes of each tool before selecting it.
- Choose ONE action: web|browser|file|math|execute|finish
- Input is concise and directly relevant to the chosen action. Be specific in your requests. When using 'web' or 'browser', specify the exact information you are seeking or the action you want to perform on the webpage. When using 'file', specify the exact file and the specific data you need. When using 'execute', provide clear and well-commented code.
- Observe results carefully and adjust your strategy accordingly. Critically evaluate the output of each tool. If the output is unexpected or incomplete, revise your input or choose a different tool.

Available tools:
- web: navigate/search/scrape via web_search/scrape_web_playwright. Use this for quick information retrieval and simple data extraction from websites. Prioritize this for fact-finding and initial exploration. Be precise in your search queries to avoid irrelevant results.
- browser: interactive browsing with UI clicks, navigation, forms, viewing PDFs online via browser_use. Employ this when web is insufficient due to interactivity requirements, complex navigation, or the need to interact with web applications. Be mindful of the page structure and use specific selectors for interacting with elements.
- file: read local files via file_io/excel_reader/pdf_reader/image_reader. Utilize this for accessing and extracting information from local files when the problem explicitly mentions them or when information is likely to be stored locally. Specify the file path and the desired data format (e.g., text, JSON, CSV).
- math: evaluate an expression via math_calculator. Use this for any calculations, no matter how simple. Double-check your input to avoid errors.
- execute: run Python via code_executor. Employ this for complex data manipulation, algorithmic tasks, or when other tools are insufficient. Be mindful of the execution time and resource limitations. Include error handling in your code to gracefully handle unexpected input or errors.
- finish: Conclude the task when you have gathered all necessary information and are confident in your final answer.

Problem:
{problem}

Context:
{context}

When you choose action finish:
 - Set <input> to ONLY the exact final answer string.
 - Do not include quotes, year, citations, or extra words.
 - For titles, return the title text only. For numbers, return only the number.
 - If the question is multiple-choice A/B/C/D, also END your overall output with exactly one line: The final answer is: X.
 - If the problem is mathematics, also append a final line: $$oxed{{<final answer only>}}$$.

Guidance:
- Prefer web for simple URL-based search/scrape.
- Use browser when interactivity is required (clicking UI, multi-step navigation, viewing PDFs inline, login/gate flows).
- Before using 'execute', consider if other tools can achieve the same result more efficiently. 'execute' should be reserved for tasks that genuinely require code execution.
- Carefully analyze the context provided. It may contain crucial information that can guide your tool selection and strategy.
- For tasks involving multiple steps, break down the problem into smaller, manageable sub-problems.
- When encountering ambiguous or incomplete information, use 'web' or 'browser' to clarify the ambiguity or find the missing information.
- If a tool fails, analyze the error message and try a different approach or tool.

Respond strictly as:
<step>
<thought>...</thought>
<action>web|browser|file|math|execute|finish</action>
<input>...</input>
</step>
"""

EARLYEXIT_PROMPT = """If the answer is already determined or unsolvable, signal early exit with a reason.

<response>
<reason><![CDATA[
Reason to exit.
]]></reason>
</response>
"""

