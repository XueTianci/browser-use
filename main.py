from pathlib import Path
from contextlib import redirect_stdout
import base64
import json
import logging
import os
import re
import sys
from typing import Any, Iterable, List
import copy
import asyncio
from pydantic import SecretStr

import cv2
import numpy as np
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether

sys.path.insert(0, "./browser_use")
from browser_use import Agent, Browser 
from browser_use import BrowserConfig 
from browser_use.browser.context import ( 
    BrowserContextConfig,
)
from browser_use.logging_config import *  
from dotenv import load_dotenv
load_dotenv()

setup_logging()
LOGGER = logging.getLogger("browser_use")

OUTPUT_DIR = Path("./result").resolve()
(OUTPUT_DIR / "image_inputs").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "trajectory").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "log").mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = OUTPUT_DIR / "process_log.txt"

SALIENT_ATTRIBUTES = (
    "alt",
    "aria-describedby",
    "aria-label",
    "aria-role",
    "aria-controls",
    "input-checked",
    "label",
    "name",
    "option_selected",
    "placeholder",
    "readonly",
    "text-value",
    "title",
    "value",
    "data-gtm-label",
    "href",
    "role",
)



def flatten_list(items: Iterable[Any]) -> List[Any]:
    """Recursively flatten *items* into a single list."""
    flat: List[Any] = []
    for itm in items:
        if isinstance(itm, list):
            flat.extend(flatten_list(itm))
        else:
            flat.append(itm)
    return flat


def process_element_tag(element: str, salient_attributes: Iterable[str]) -> str:
    """Clean an HTML *element* string, keeping only *salient_attributes*."""
    if not element.endswith(">"):
        element += "'>"

    soup = BeautifulSoup(element, "html.parser")
    for tag in soup.find_all(True):
        # Keep only wanted attributes
        filtered_attrs = {k: tag.attrs[k] for k in tag.attrs if k in salient_attributes}
        name_val = filtered_attrs.pop("name", None)
        new_tag = soup.new_tag(tag.name, **filtered_attrs)
        if name_val:
            new_tag["name"] = name_val
        return str(new_tag).split(f"</{tag.name}>")[0]
    return element


def decode_and_save_image(b64_img: str, path: Path) -> None:
    """Decode base64 *b64_img* and write to *path*."""
    image_array = np.frombuffer(base64.b64decode(b64_img), np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    cv2.imwrite(str(path), img)


def save_screenshots(
    screenshots: list[str],
    subdir: Path,
    prefix: str,
) -> None:
    for count, shot in enumerate(screenshots):
        if shot == None:
            continue
        decode_and_save_image(shot, subdir / f"{count}_{prefix}.png")

reasoning_models = ["o4-mini-2025-04-16", "o3-2025-04-16"]

def initialize_llm(kwargs):
    model_name = kwargs["model_name"]
    reasoning_effort = kwargs.get("reasoning_effort", "medium")

    if model_name in reasoning_models:
        print(f"Initializing {model_name} with reasoning effort: {reasoning_effort}")
        return ChatOpenAI(model=model_name, reasoning_effort=reasoning_effort)
    elif "gemini" in model_name:
        print(f"Initializing Gemini model: {model_name}")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    elif "claude" in model_name:
        if reasoning_effort == "high":
            print(f"Initializing Claude with high reasoning effort")
            return ChatAnthropic(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4096 + 4096,
                thinking={"type": "enabled", "budget_tokens": 4096},
            )
        return ChatAnthropic(model=model_name, temperature=0)
    elif "deepseek" in model_name:
        print(f"Initializing DeepSeek model: {model_name}")
        return ChatTogether(model=model_name, temperature=0)
    else:
        print(f"Initializing default OpenAI model: {model_name}")
        return ChatOpenAI(model=model_name, temperature=0)


def initialize_agent(task, llm, browser, kwargs):
    model_name = kwargs["model_name"]
    if "deepseek" in model_name:
        use_vision = False
    else:
        use_vision = True
    if model_name in reasoning_models and "claude" not in model_name:
        json_mode = True
        print(f"Initializing agent with use_vision={use_vision}, json_mode={json_mode}")
        return Agent(
            task=task["confirmed_task"],
            llm=llm,
            browser=browser,
            use_vision=use_vision,
            save_conversation_path=str(OUTPUT_DIR / "log/conversation.json"),
            tool_calling_method="json_mode",
        )
    else:
        json_mode = False
        print(f"Initializing agent with use_vision={use_vision}, json_mode={json_mode}")
        return Agent(
            task=task["confirmed_task"],
            llm=llm,
            browser=browser,
            use_vision=use_vision,
            save_conversation_path=str(OUTPUT_DIR / "log/conversation.json"),
        )


async def run_single_task(task: dict[str, str], **kwargs: Any) -> None:
    """Run the browser agent on a single task contained in *input*."""
    llm = initialize_llm(kwargs)

    file_handler = logging.FileHandler(LOG_FILE_PATH)
    LOGGER.addHandler(file_handler)

    with LOG_FILE_PATH.open("w") as log_file, redirect_stdout(log_file):
        try:
            # Basic configuration for browser
            config = BrowserConfig(
                headless=False,
                disable_security=True,
                new_context_config=BrowserContextConfig(initial_url=task["website"]),
            )

            browser = Browser(config=config)

            agent = initialize_agent(task, llm, browser, kwargs)

            result = await agent.run(max_steps=kwargs["max_steps"])

            result_dict = copy.deepcopy(task)
            result_dict["urls"] = result.urls()
            result_dict["errors"] = result.errors()
            result_dict["model_actions"] = result.model_actions()
            result_dict["extracted_content"] = result.extracted_content()
            result_dict["final_result_response"] = result.final_result()
            result_dict["action_names"] = result.action_names()
            result_dict["action_results"] = result.action_results()
            result_dict["model_thoughts"] = result.model_thoughts()

            clean_extracted_content = []

            if result_dict["action_names"] and result_dict["action_names"][-1] == "done":
                temp = result_dict["extracted_content"][:-1]
                result_dict["final_result_response"] = result_dict["extracted_content"][-1]
            else:
                temp = result_dict["extracted_content"]
                result_dict["final_result_response"] = ""

            for content in temp:
                    cleaned_content = re.sub(r'\\u[a-fA-F0-9]{4}|\\U[a-fA-F0-9]{8}', '', content.encode('unicode_escape').decode('utf-8')).strip()
                    if not cleaned_content.startswith("Extracted page content"):
                        if "->" in cleaned_content:
                            element = cleaned_content.split("->")[0].strip()
                            op = cleaned_content.split("->")[1].strip()
                            filtered_element = str(process_element_tag(element, SALIENT_ATTRIBUTES))
                            final_action = filtered_element + " -> " + op
                            clean_extracted_content.append(final_action)
                        else:
                            clean_extracted_content.append(cleaned_content)
                    else:
                        clean_extracted_content.append("Extracted page content")

            result_dict["action_history"] = clean_extracted_content

            save_screenshots(result.screenshots(), OUTPUT_DIR / "image_inputs", "screenshot")
            save_screenshots(result.ori_screenshots(), OUTPUT_DIR / "trajectory", "screenshot")
            # Write structured results

            with open(f"{OUTPUT_DIR}/result.json", "w") as f:
                json.dump(result_dict, f)

        except Exception as exc:
            print(f"An error occurred: {exc}")

    LOGGER.removeHandler(file_handler)


def run(input: dict[str, dict], **kwargs: Any) -> dict[str, str]:
    assert 'model_name' in kwargs, 'model_name is required'
    assert len(input) == 1, 'input must contain only one task'

    print(kwargs)
    
    task_id, task = list(input.items())[0]
    task["task_id"] = task_id
    print(f"Task ID: {task_id}")
    print(f"Task: {task}")

    asyncio.run(run_single_task(task, **kwargs))
    
    results = {}
    results[task_id] = {"trajectory": f"./results/online_mind2web/RUN_ID/{task_id}/result/trajectory", 
                        "result_file": f"./results/online_mind2web/RUN_ID/{task_id}/result/result.json"}
    return results