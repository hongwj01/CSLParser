from langchain_openai import ChatOpenAI
import json

MODEL = ""
API_KEY = ""
BASE_URL = ""


class LLMForLogParsing:
    def __init__(self):
        pass

    def parse_log_complex(self, log, template, confidences, examples):
        prompt_template = """
Your task is to perform log parsing and template generation. Logs are unstructured or semi-structured text data that often contain useful information in a standardized format. The goal is to extract key variables from logs, and ensure logs are structured for downstream analysis. You must identify and abstract dynamic variables in logs with `<*>` (one variable corresponds to one `<*>`) and output a static log template.
    
We need to review the parsed log parameters in a complex log template. Due to the complexity of the template, parsing may be inaccurate, particularly with mismatched brackets or quotes that might have been incorrectly treated as part of variables. Refer to examples to decide whether such symbols should be grouped as part of a variable or extracted separately for consistency. Provide accurate and concise results.

Examples log template: 
{examples}

Log to analyze: 
{log}

Small model parsing results:
{template}

Small model parsing confidence for varibles:
{confidences}

Your response must be in structured JSON format with the following keys:
- `parsed_template` (str): The resulting log template with placeholders for variable content.
    """

        examples_str = "\n".join(
            [
                f"Example {i+1}:\n"
                f"**Log**: {example['log']}\n"
                f"**Ground Truth Template**: {example['template']}\n"
                for i, example in enumerate(examples)
            ]
        )

        final_prompt = prompt_template.format(
            log=log, template=template, confidences=confidences, examples=examples_str)

        chat = ChatOpenAI(
            model=MODEL,
            temperature=0,
            seed=0,
            max_tokens=1024,
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL
        )
        try:
            response = chat.invoke(
                [
                    {"role": "system", "content": "You are an expert in log parsing."},
                    {"role": "user", "content": final_prompt}
                ]
            )
        except Exception as e:
            print("LLM Server Error.")
            return "", {'total_tokens': 0}

        try:
            result = response.content.strip()
            result = str(result)
            result = result.split('```json')[-1].split('```')[0]
            result_json = json.loads(result)
            return result_json['parsed_template'], response.usage_metadata
        except Exception as e:
            return "", response.usage_metadata

    def parse_log_keyword(self, log, template, confidences, examples):
        prompt_template = """
Your task is to perform log parsing and template generation. Logs are unstructured or semi-structured text data that often contain useful information in a standardized format. The goal is to extract key variables from logs, and ensure logs are structured for downstream analysis. You must identify and abstract dynamic variables in logs with `<*>` (one variable corresponds to one `<*>`) and output a static log template.
    
We have a log entry that contains important keywords (e.g., exception types), which are crucial for understanding the log's context, so these keywords and specific exception details must be retained during parsing. The small model has parsed the log, but may have parsed some keywords into variables.  Your task is to focus **only on the parts of the log that have already been abstracted as <*> by the small model** and ensure that both the keywords and the specific error details are preserved accurately.

Example log template: 
{examples}

Log to analyze: 
{log}

Small model parsing results:
{template}

Small model parsing confidence for varibles:
{confidences}

**Important Rules:**
**Prioritize Ground Truth Templates: If a similar example exists, use its ground truth template.**
**Preserve Keywords and Constants: Do not modify fixed text or keywords in the log. Only abstract dynamic variables as <*>.**
Example 1: 
Log: "www.example.com:80 error unable to process request: java.io.IOException: File not found"
Small Model Output: "<*> error unable to process request: <*>: <*>"
Correct Template: "<*> error unable to process request: java.io.IOException: File not found"
Reason: Preserve critical semantic details like exception type (java.io.IOException) and cause (File not found). Abstract only dynamic parts (e.g., URL).
Example 2: 
Log: "[<12345678>] handle_data_transfer+0xAB/0xCD"
Small Model Output: "[<*>] <*>"
Correct Template: "[<*>] handle_data_transfer+<*>"
Reason: Retain function name (handle_data_transfer) for context; abstract only dynamic offsets (0xAB/0xCD).

Your response must be in structured JSON format with the following keys:
- `parsed_template` (str): The resulting log template with placeholders for variable content.
    """

        examples_str = "\n".join(
            [
                f"Example {i+1}:\n"
                f"**Log**: {example['log']}\n"
                f"**Ground Truth Template**: {example['template']}\n"
                for i, example in enumerate(examples)
            ]
        )

        final_prompt = prompt_template.format(
            log=log, template=template, confidences=confidences, examples=examples_str)

        chat = ChatOpenAI(
            model=MODEL,
            temperature=0,
            seed=0,
            max_tokens=1024,
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL
        )
        try:
            response = chat.invoke(
                [
                    {"role": "system", "content": "You are an expert in log parsing."},
                    {"role": "user", "content": final_prompt}
                ]
            )
        except Exception as e:
            print("LLM Server Error.")
            return "", {'total_tokens': 0}

        try:
            result = response.content.strip()
            result = str(result)
            result = result.split('```json')[-1].split('```')[0]
            result_json = json.loads(result)
            return result_json['parsed_template'], response.usage_metadata
        except Exception as e:
            return "", response.usage_metadata

    def parse_log_confidence(self, log, template, confidences, examples):
        prompt_template = """
Your task is to perform log parsing and template generation. Logs are unstructured or semi-structured text data that often contain useful information in a standardized format. The goal is to extract key variables from logs, and ensure logs are structured for downstream analysis. You must identify and abstract dynamic variables in logs with `<*>` (one variable corresponds to one `<*>`) and output a static log template.
    
We have a log entry that has been parsed by a small model, but the small model's parsing results come with varying levels of confidence. Specifically, some parameters have a low confidence, which may result in errors or imprecise parsing. Your task is to focus **only on the parts of the log that have already been abstracted as <*> by the small model**, especially those with low confidence (e.g., below 0.7), and re-evaluate whether these abstractions are correct.

Example log template: 
{examples}

Log to analyze: 
{log}

Small model parsing results:
{template}

Small model parsing confidence for varibles:
{confidences}

**Important Rules:**
**Prioritize Ground Truth Templates: If a similar example exists, use its ground truth template.**
**Preserve Keywords and Constants: Do not modify fixed text or keywords in the log. Only abstract dynamic variables as <*>.**
Example 1: Keeping Variables as <*>
If the small model correctly abstracts a dynamic variable as <*>, leave it unchanged.
Log: "service_abc startup succeeded."
Correct Template: "<*> startup succeeded."
Reason: service_abc is dynamic (e.g., service name), so <*> is correct. No changes needed.
Example 2: Restoring Constants
If the small model incorrectly replaces a constant with <*>, restore the original constant.
Log: "ERROR Unable to process request: java.io.IOException."
Parsed: "ERROR Unable to process request: <*>."
Correct Template: "ERROR Unable to process request: java.io.IOException."
Reason: java.io.IOException is a specific exception type (constant) and should not be abstracted.

Your response must be in structured JSON format with the following keys:
- `parsed_template` (str): The resulting log template with placeholders for variable content.
    """

        examples_str = "\n".join(
            [
                f"Example {i+1}:\n"
                f"**Log**: {example['log']}\n"
                f"**Ground Truth Template**: {example['template']}\n"
                for i, example in enumerate(examples)
            ]
        )

        final_prompt = prompt_template.format(
            log=log, template=template, confidences=confidences, examples=examples_str)

        chat = ChatOpenAI(
            model=MODEL,
            temperature=0,
            seed=0,
            max_tokens=1024,
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL
        )
        try:
            response = chat.invoke(
                [
                    {"role": "system", "content": "You are an expert in log parsing."},
                    {"role": "user", "content": final_prompt}
                ]
            )
        except Exception as e:
            print("LLM Server Error.")
            return "", {'total_tokens': 0}

        try:
            result = response.content.strip()
            result = str(result)
            result = result.split('```json')[-1].split('```')[0]
            result_json = json.loads(result)
            return result_json['parsed_template'], response.usage_metadata
        except Exception as e:
            return "", response.usage_metadata

    def merge_template(self, old_template, old_log, new_template, new_log, distance):
        prompt_template = """
Your task is to analyze two log templates and determine whether they should be merged into a single, more general template. Logs are unstructured or semi-structured text data that often contain useful information in a standardized format. The goal is to extract key variables from logs, abstract dynamic variables with `<*>` (one variable corresponds to one `<*>`), and output a static log template.

Guidelines for merging:
1. **Semantic Similarity**: Ensure that the two templates express the same event or concept. For example:
   - "API response status=<*>" and "API response result=<*>" might look similar structurally, but they describe different outcomes (successful response vs. a result) and should not be merged.
   - "Exception java.io.EOFException" and "Exception java.nio.channels.ClosedByInterruptException" might share some structure, but they represent distinct exceptions and should not be merged.
   - "Exception java.net.ConnectException: Connection timed out" and "Exception java.net.ConnectException: Connection refused" might involve the same exception type (java.net.ConnectException), but the specific error messages after the colon describe different issues and should not be merged.
2. **Variable Abstraction**: Only abstract dynamic parts (variables) as `<*>`. Fixed tokens that carry semantic meaning must remain unchanged. For example:
   - "User authentication failed for alice" and "User authentication failed for root" can be merged as "User authentication failed for <*>".
   - "File upload completed: /path/to/file" and "File upload completed: /another/path" can be merged as "File upload completed: <*>".
   - Do not modify fixed tokens like "authentication", "upload", etc., as they carry important semantic information.
3. **Matching Capability**: The merged template must be able to match all logs that originally matched either of the two input templates. For example:
   - If the original templates are "User authentication failed for alice" and "User authentication failed for root", the merged template "User authentication failed for <*>" should match both "User authentication failed for alice" and "User authentication failed for root".
   - If the merged template cannot match all original logs, it is invalid and the templates should not be merged.

Input:
template_1: "{old_template}"
log_1: "{old_log}"
template_2: "{new_template}"
log_2: "{new_log}"
distance: {distance}

Your response must be in structured JSON format with the following keys:
- `flag` (str): Indicates whether the two templates can be merged. Use `'Merge'` if they can be merged, and `'No'` if they cannot.
- `template` (str): The resulting merged log template if `flag` is `'Merge'`. If `flag` is `'No'`, this field should be `null`.
"""

        final_prompt = prompt_template.format(
            old_template=old_template, old_log=old_log, new_template=new_template, new_log=new_log, distance=distance)

        chat = ChatOpenAI(
            model=MODEL,
            temperature=0,
            seed=0,
            max_tokens=1024,
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL
        )
        try:
            response = chat.invoke(
                [
                    {"role": "system", "content": "You are an expert in log parsing."},
                    {"role": "user", "content": final_prompt}
                ]
            )
        except Exception as e:
            print("LLM Server Error.")
            return None, None, {'total_tokens': 0}

        try:
            result = response.content.strip()
            result = str(result)
            result = result.split('```json')[-1].split('```')[0]
            result_json = json.loads(result)
            return result_json['flag'], result_json['template'], response.usage_metadata
        except Exception as e:
            return None, None, response.usage_metadata
