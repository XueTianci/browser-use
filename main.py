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

import cv2
import numpy as np
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI, AzureChatOpenAI

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
    urls: list[str],
    screenshots: list[str],
    subdir: Path,
    prefix: str,
) -> None:
    """Save *screenshots* whose corresponding *urls* are valid into *subdir*."""
    for count, (url, shot) in enumerate(zip(urls, screenshots)):
        if "chrome-error" in url or url == "about:blank":
            continue
        decode_and_save_image(shot, subdir / f"{count}_{prefix}.png")


async def run_single_task(task: dict[str, str], **kwargs: Any) -> None:
    """Run the browser agent on a single task contained in *input*."""

    llm = ChatOpenAI(model=kwargs["model_name"], temperature=0)

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
            agent = Agent(
                task=task["confirmed_task"],
                llm=llm,
                browser=browser,
                use_vision=True,
                save_conversation_path=str(OUTPUT_DIR / "log/conversation.json"),
            )

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

            for content in result_dict["extracted_content"][:-1]:
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

            save_screenshots(result.urls(), result.screenshots(), OUTPUT_DIR / "image_inputs", "screenshot")
            save_screenshots(result.urls(), result.ori_screenshots(), OUTPUT_DIR / "trajectory", "screenshot")
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

    # asyncio.run(run_single_task(task, **kwargs))
    
    results = {}
    results[task_id] = {"trajectory": f"./results/online_mind2web/RUN_ID/{task_id}/result/trajectory", 
                        "result_file": f"./results/online_mind2web/RUN_ID/{task_id}/result/result.json"}
    return results