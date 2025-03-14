from sympy import *
from sympy.parsing.latex import parse_latex
import re
from wrapt_timeout_decorator import *
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk.tokenize import word_tokenize
from typing import Callable, Dict,Any, List, Optional, Tuple
import random
import pandas as pd
import copy
import signal
from openai import OpenAI

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string
def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        ss1 = parse_latex(ss1)
        ss2 = parse_latex(ss2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception as e:
        return str1 == str2

def extract_via_gpt(input_str: str,question:str) -> str:
    """Extracts letter from answer.

    Args:
        input_str : answer string - answers should start as mentioned in starts_with_capital_letter.

    Returns:
        letter or "-1".
    """
    client = OpenAI(
        api_key="sk-8eHzwMgmpm05NTJ6y26e2SiWbAPf6wLGIiH2zr0u1iHjZd1p",
        base_url="https://api.claudeshop.top/v1"
    )

    message = [{
        'role': 'user',
        'content': "Here is a model's answer about following question.\nQuestion:{}\n Please extract the EXACT answer (only one capital letter) from the answer text as the final answer for question.\n\n[Example]: When a picture frame is not hung vertically, placing it on a flat surface is a practical solution. A table provides a stable and accessible location for displaying the frame, allowing for easy viewing and enhancing the decor of the space. Other options, such as a wall or a car, may not offer the same stability or visibility. Therefore, the best choice remains: E: table\n[Extracted Answer]: E\n\n[Example]: The correct answer is A because it aligns with the provided evidence. After further analysis, the answer is still A.\n[Extracted Answer]: A\n\n[Example]: The answer is B. The other options are not supported by the evidence.\n[Extracted Answer]: B\n\n[Example]: To analyze the question, we need to consider the context in which a man would take paperwork to consult with others. 1. **Desk**: This is typically a personal workspace and not a place where multiple people gather to consult over paperwork. It seems less likely. 2. **Meeting**: This is a formal gathering where people discuss specific topics, often involving paperwork. This option fits well with the idea of consulting over paperwork. 3. **Office**: This is a general term for a workplace. While it could involve consulting, it doesn't specifically imply a gathering of people for discussion. 4. **Table**: This is a piece of furniture and doesn't inherently suggest a location for consultation. It lacks context. 5. **Work**: This is a broad term that refers to a place or activity. It does not specifically indicate a location for consulting over paperwork. Given this analysis, the most fitting answer is **B: meeting**, as it directly implies a gathering of people to discuss paperwork.\n[Extracted Answer]: B\n\n[Example]: The correct answer is A.\n[Extracted Answer]:A\n\n[Example]: A.key The correct answer is key.\n[Extracted Answer]:A\n\n[Example]: Answer: C is justified by the information given in the previous section.\n[Extracted Answer]:C\n\n[Example]: (D)\n[Extracted Answer]:D\n\n[Example]: {}\n[Extracted Answer]:".format(question,input_str)    }]

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=message
    ).to_dict()
    response = response["choices"][0]["message"]["content"]
    if len(response)<1:
        return "-1"
    else:
        return response[0]

def extract_last_num(text: str) -> str:
        match = re.search(r'boxed\{(.*?)\}', text)
        if match:
            text=match.group(1)
        text = re.sub(r"(\d),(\d)", r"\1\2", text)  
        res = re.findall(r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text)
        if len(res) > 0:
            num_str = res[-1]
            return num_str
        else:
            return "Error"

def most_frequent_by_bleu(list: List[str]) -> Tuple[str, int]:
    best_score=0
    best_code=list[0]
    smooth_func = SmoothingFunction().method1
    for i,current_code in enumerate(list):
        current_score=0
        for index,code in enumerate(list):
            if i==index:continue
            # current_score+=sacrebleu.corpus_bleu(word_tokenize(current_code), [word_tokenize(code)], lowercase=True).score
            current_score+=sentence_bleu([word_tokenize(code)], current_code,smoothing_function=smooth_func)
            # if current_score==0:
            #     print(word_tokenize(current_code))
            #     print(word_tokenize(code))
            #     time.sleep(10)
        if current_score>best_score:
            best_score=current_score
            best_code=current_code
    return best_code,best_score

def extract_final_answers_gsm8k(outputs):
    # need to consider robostness
    final_answers = []
    for output in outputs:
            final_answer = extract_last_num(output)
            final_answers.append(final_answer)
    return final_answers

def most_frequent(ans_list):
    # need to consider robostness
    if len(ans_list)==0:
        return "-1"
    counter = 0
    num = ans_list[0]

    for i in ans_list:
        current_frequency = ans_list.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

def get_boxed_answer(solutions):
    final_answers=[]
    for solution in solutions:
        tmp = solution.split('oxed{')[-1]
        count = 0
        answer = ''
        for item in tmp:
            if item == '}':
                if count == 0:
                    break
                else:
                    count -=1
            elif item == '{':
                count += 1
            answer += item
        final_answers.append(answer)
    return final_answers

def get_last_uppercase(solutions):
    final_answers=[]
    for solution in solutions:
        match = re.findall(r'[A-J]', solution)  

        if match:
            final_answers.append(match[0])
        else:
            final_answers.append("")
    return final_answers

def latex_to_python(latex_expr):
    """
    将 LaTeX 格式的加减乘除符号转换为 Python 运算符。
    :param latex_expr: str, 包含 LaTeX 表达式的字符串
    :return: str, Python 运算符组成的表达式字符串
    """
    # 替换 LaTeX 符号为 Python 运算符
    latex_expr = latex_expr.replace('\\+', '+')  # 加号
    latex_expr = latex_expr.replace('\\-', '-')  # 减号
    latex_expr = latex_expr.replace('\\times', '*')  # 乘号
    latex_expr = latex_expr.replace('\\cdot', '*')  # 乘号（点符号）
    latex_expr = latex_expr.replace('\\div', '/')  # 除号
    latex_expr = latex_expr.replace('\\frac', '/')  # 分式
    return latex_expr

def extract_final_answers_game24(outputs):
    # need to consider robostness
    final_answers = []
    for output in outputs:
            final_answer = extract_last_num(output)
            final_answers.append(final_answer)
    return final_answers



def validate_expression(numbers, expression):
    # 检查等式是否为24
    try:
        if eval(expression.split('=')[0]) != 24:
            return False
    except:
        return False

    # 提取等式中的数字
    import re
    expression_numbers = re.findall(r'\d+', expression.split('=')[0])
    expression_numbers = list(map(int, expression_numbers))
    # print(expression_numbers)
    # 检查输入的数字是否全部使用且只使用一次
    return sorted(numbers) == sorted(expression_numbers)

# 3.humaneval, mbpp代码提取
def extract_code_answer(text: str) -> str:

    text = text.split('```python')[-1]
    text = text.split('```')[0]
    if len(text) > 0:
        text = text.replace("print(", "#print(")
        return text
    else:
        return "NONE"
    
# 4.humaneval代码评测
@timeout(20)
def format_solution_humaneval(answer: str, target: str) ->bool:
    '''
    match_fn(answer, target)
    Args:
    answer: codes from agents
    target : solution: dict
    '''
    eval_code = answer + '\n' + target["test"] + f'\ncheck({target["entry_point"]})'
    exec_globals = {}
    try:
        exec(eval_code,exec_globals)
        return True
    except:
        return False
    
# 5.mbpp代码评测
@timeout(20)
def format_solution_mbpp(answer: str, target: str) ->bool:
    '''
    match_fn(answer, target)
    Args:
    answer: codes from agents
    target : solution: dict
    '''
    answer = '\n'+answer
    pos = answer.rfind('\ndef ')
    last_func_name = answer[pos+5:].split('(')[0]
    format_answer = answer.replace(last_func_name, target["func_name"])
    eval_code = ""
    for item in target["test_setup_code"]:
        eval_code += item
    eval_code += format_answer
    try:
        exec(eval_code, globals())
        for test in target["test_list"]:
            exec(test)
        return True
    except:
        return False
def compute_correctness(ans_list,s,dataset):
    is_correct=0
    if dataset not in ["humaneval","mbpp"]: 
        ans_list=[item for item in ans_list if item !="Error" and item!=""]
        ans_list=[item.replace(" ","") for item in ans_list]
        ans= most_frequent(ans_list)
    else:
        ans,_=most_frequent_by_bleu(ans_list)
    if dataset in ["gsm8k","MGSM"]:
        a = float(ans.replace(',',''))
        if dataset=="MGSM":
            s=float(s["solution"])
        else:
            s=float(s["solution"].replace(',',''))
        if abs(s-a) < 1e-6:
            is_correct=1
    elif dataset=="math":
        if is_equiv(s["solution"].replace(" ",""),ans):
            is_correct=1
    elif dataset =="humaneval":
        if format_solution_humaneval(ans,s):
            is_correct=1
    elif dataset == "mbpp":
        if format_solution_mbpp(ans,s):
            is_correct=1
    elif dataset == "logiqa":
        matches = re.findall(r'\{(.*?)\}', ans)
        if matches:
            ans=matches[0]
        if ans.replace(" ","").lower()==chr(65+s["solution"]).replace(" ","").lower():
            is_correct=1
    else:
        matches = re.findall(r'\{(.*?)\}', ans)
        if matches:
            ans=matches[0]
        if ans.replace(" ","").lower()==s["solution"].replace(" ","").lower():
            is_correct=1

    
    return is_correct
def calculate_num_sampling(ans_list):
    set_list=set(ans_list)
    num_list=[]
    for set_ans in set_list:
        num_list.append(ans_list.count(set_ans))
    return max(num_list)/len(ans_list)