import os
import requests
import nbformat as nbf
from jupyter_client.manager import start_new_kernel
from queue import Empty
import json
import base64
import re
import traceback
import logging

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from openai import OpenAI

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

import uvicorn
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from ollama import AsyncClient

client = AsyncClient()
MODEL = "llama3.2:3b"    

app = FastAPI()

class UserQuestion(BaseModel):
    question: str
    notebook_filename: Optional[str] = "data_analysis_notebook.ipynb"
    file_path: Optional[str] = None  # e.g., "titanic.csv"
    allow_dangerous_code: Optional[bool] = False  # Whether to allow code that modifies the file system or installs packages

# Class and function definitions

class NotebookSession:
    def __init__(self, notebook_filename, executed_cells_filename):
        # Get the directory where main.py is located
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create full paths for notebook and execution state files
        self.notebook_filename = os.path.join(self.base_dir, notebook_filename)
        self.executed_cells_filename = os.path.join(self.base_dir, executed_cells_filename)

        logger.info("Initializing NotebookSession")
        logger.info(f"Notebook filename: {self.notebook_filename}")
        logger.info(f"Executed cells filename: {self.executed_cells_filename}")

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.notebook_filename), exist_ok=True)

        # Load notebook state from file or initialize a new notebook
        self.nb = self.load_notebook_state()
        # Load executed cells from file or initialize an empty set
        self.executed_cells = self.load_execution_state()

        # Start a new Jupyter kernel for the session
        self.km, self.kc = start_new_kernel(kernel_name="python3")
        logger.info("Jupyter kernel started")

    def load_notebook_state(self):
        """
        Loads the notebook state from a file or creates a new notebook if the file doesn't exist.
        """
        if os.path.exists(self.notebook_filename) and os.path.getsize(self.notebook_filename) > 0:
            logger.info("Loading existing notebook state")
            with open(self.notebook_filename, 'r') as f:
                return nbf.read(f, as_version=4)
        else:
            logger.info("No existing notebook found, creating a new one")
            return nbf.v4.new_notebook()

    def save_notebook_state(self):
        """
        Saves the current notebook state to a file.
        """
        with open(self.notebook_filename, 'w') as f:
            nbf.write(self.nb, f)
        logger.info("Notebook state saved")

    def load_execution_state(self):
        """
        Loads the executed cells state from a file or initializes an empty set if the file doesn't exist.
        """
        if os.path.exists(self.executed_cells_filename) and os.path.getsize(self.executed_cells_filename) > 0:
            logger.info("Loading execution state")
            with open(self.executed_cells_filename, 'r') as f:
                return set(json.load(f))
        else:
            logger.info("No existing execution state found, initializing an empty set")
            return set()

    def save_execution_state(self):
        """
        Saves the executed cells state to a file.
        """
        with open(self.executed_cells_filename, 'w') as f:
            json.dump(list(self.executed_cells), f)
        logger.info("Execution state saved")

    def run_code_in_kernel(self, code):
        """
        Runs a code string in the current kernel session and returns the output and any errors.
        """
        logger.debug(f"Running code in kernel:\n{code}")
        msg_id = self.kc.execute(code)
        output = []
        error_output = None
        image_counter = 0

        # Flags to track when execution is complete
        execute_reply_received = False

        while True:
            try:
                # Process messages on the IOPub channel
                msg = self.kc.get_iopub_msg(timeout=1)
                msg_type = msg['msg_type']
                if msg_type == 'stream':
                    output.append(msg['content']['text'])
                elif msg_type == 'execute_result':
                    data = msg['content']['data']
                    if 'text/plain' in data:
                        output.append(data['text/plain'])
                    elif 'text/html' in data:
                        output.append(self.strip_html_tags(data['text/html']))
                elif msg_type == 'display_data':
                    data = msg['content']['data']
                    if 'text/plain' in data:
                        output.append(data['text/plain'])
                    elif 'text/html' in data:
                        output.append(self.strip_html_tags(data['text/html']))
                    if 'image/png' in data:
                        image_data = data['image/png']
                        image_filename = f"output_image_{image_counter}.png"
                        full_image_path = os.path.join(self.base_dir, image_filename)
                        with open(full_image_path, "wb") as f:
                            f.write(base64.b64decode(image_data))
                        output.append(f"Image saved as {image_filename}")
                        image_counter += 1
                elif msg_type == 'error':
                    error_output = 'Error: ' + ''.join(msg['content']['traceback'])
                    output.append(error_output)
            except Empty:
                # No more messages on IOPub channel at the moment
                pass

            # Check if the execute_reply message has been received
            if not execute_reply_received:
                try:
                    reply = self.kc.get_shell_msg(timeout=0)
                    if reply['parent_header'].get('msg_id') == msg_id:
                        execute_reply_received = True
                        if reply['content']['status'] == 'error':
                            error_output = 'Error: ' + ''.join(reply['content']['traceback'])
                except Empty:
                    # No execute_reply yet, continue waiting
                    pass

            # Break the loop if execution is complete
            if execute_reply_received:
                break

        logger.debug(f"Kernel output:\n{''.join(output)}")
        if error_output:
            logger.error(f"Kernel error output:\n{error_output}")

        return ''.join(output), error_output

    def strip_html_tags(self, html_content):
        """
        Utility function to strip HTML tags from the content.
        """
        # Using regex to strip HTML tags
        clean_text = re.sub('<.*?>', '', html_content)
        return clean_text

    def add_cell(self, new_code):
        """
        Adds a new code cell to the notebook but does not run it yet.
        """
        logger.info("Adding a new cell to the notebook")
        logger.debug(f"New code:\n{new_code}")
        new_code_cell = nbf.v4.new_code_cell(new_code)
        self.nb['cells'].append(new_code_cell)
        # Save the notebook state
        self.save_notebook_state()
        # Return the index of the new cell
        logger.info(f"Cell added at index {len(self.nb['cells']) - 1}")
        return f"Cell added at index {len(self.nb['cells']) - 1}."

    def edit_cell(self, cell_index, new_code):
        """
        Edits the cell at the given index and replaces it with new_code but does not run it yet.

        Parameters:
        - cell_index: The index of the cell to be replaced (starting from 0).
        - new_code: The new code to replace the existing cell content.
        """
        logger.info(f"Editing cell at index {cell_index}")
        if 0 <= cell_index < len(self.nb['cells']):
            logger.debug(f"New code for cell {cell_index}:\n{new_code}")
            self.nb['cells'][cell_index]['source'] = new_code
            # Remove from executed_cells to ensure it gets re-executed
            if cell_index in self.executed_cells:
                self.executed_cells.remove(cell_index)
            # Save the notebook state
            self.save_notebook_state()
            # Save the execution state
            self.save_execution_state()
            logger.info(f"Cell at index {cell_index} edited")
            return f"Cell at index {cell_index} edited."
        raise ValueError(f"Error: Cell at index {cell_index} not found.")

    def remove_cell(self, cell_index):
        """
        Removes the cell at the given index.

        Parameters:
        - cell_index: The index of the cell to be removed (starting from 0).
        """
        if 0 <= cell_index < len(self.nb['cells']):
            # Remove the cell
            self.nb['cells'].pop(cell_index)
            # Adjust the executed_cells set
            self.executed_cells = {idx for idx in self.executed_cells if idx != cell_index}
            # Adjust indices in executed_cells set
            self.executed_cells = {idx - 1 if idx > cell_index else idx for idx in self.executed_cells}
            # Save the notebook state
            self.save_notebook_state()
            # Save the execution state
            self.save_execution_state()
            logger.info(f"Cell at index {cell_index} removed.")
            return f"Cell at index {cell_index} removed."
        else:
            raise ValueError(f"Error: Cell at index {cell_index} not found.")

    def run_cell(self, cell_index):
        """
        Runs cells up to and including the specified cell by its index.
        """
        logger.info(f"Running cell at index {cell_index}")
        if not self.nb['cells']:
            raise ValueError("Error: No cells in the notebook to run.")

        if 0 <= cell_index < len(self.nb['cells']):
            error_output = None
            # Run any unexecuted cells up to and including the target cell
            for idx in range(cell_index + 1):
                cell = self.nb['cells'][idx]
                logger.debug(f"Executing cell {idx}:\n{cell['source']}")
                _, cell_error = self.run_code_in_kernel(cell['source'])
                self.executed_cells.add(idx)
                if cell_error:
                    error_output = cell_error
                    logger.error(f"Error executing cell {idx}: {cell_error}")
                    break  # Stop execution on error
            # Save the execution state
            self.save_execution_state()
            if error_output:
                return '', error_output
            else:
                # Get the output of the target cell
                cell_output, _ = self.run_code_in_kernel(self.nb['cells'][cell_index]['source'])
                logger.info(f"Cell {cell_index} executed successfully")
                return cell_output, None
        else:
            raise ValueError(f"Error: Cell at index {cell_index} not found.")

    def get_notebook(self):
        """
        Returns the current notebook as a JSON string.
        """
        return nbf.writes(self.nb)

    def write_notebook_to_file(self):
        """
        Writes the current notebook to a file.
        """
        with open(self.notebook_filename, 'w') as f:
            nbf.write(self.nb, f)
        logger.info("Notebook written to file")

# LLM Request to generate notebook code
def LLM_Request(user_message, notebook_state, file_path=None, feedback=None):
    """
    Send a request to OpenAI to convert the user's message into Python code.
    """
    logger.info("Sending request to LLM")
    logger.debug(f"User message: {user_message}")
    
    # Parse the notebook state into a dictionary
    if notebook_state.strip() == '':
        notebook_state_dict = {'cells': []}
    else:
        notebook_state_dict = json.loads(notebook_state)

    # Extract notebook context
    notebook_overview = "\n".join(
        [f"Cell {idx}:\n{cell['source']}" for idx, cell in enumerate(notebook_state_dict['cells'])]
    )

    # Create the prompt
    prompt = f"""User's message: {user_message}
Notebook state:
{notebook_overview}
Number of cells in the notebook: {len(notebook_state_dict['cells'])}
"""

    if feedback:
        prompt += f"\nPrevious feedback: {feedback}\n"
    if file_path:
        prompt += f"\nData file location: {file_path}\n"

    # Create the messages list for the API call
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that generates and corrects Python code for a Jupyter notebook. "
                "Your response must be only in the following JSON format, enclosed within <JSON> and </JSON> tags:\n"
                "<JSON>\n"
                "{\n"
                "  \"action\": \"add\", \"edit\", or \"remove\",\n"
                "  \"cell_index\": integer or null,\n"
                "  \"code\": \"code to add or edit\",\n"
                "  \"needs_confirmation\": true or false\n"
                "}\n"
                "</JSON>"
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        # response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=messages,
        #     temperature=0.0
        # )
        response = client.chat(
                model=MODEL,
                messages=messages,
                options={'temperature': 0.0}
            )
        logger.info("LLM request successful")
        
        # Extract the JSON response
        generated_response = response.choices[0].message.content
        json_match = re.search(r'<JSON>\s*(\{.*?\})\s*</JSON>', generated_response, re.DOTALL)
        
        if json_match:
            json_text = json_match.group(1)
            response_dict = json.loads(json_text)
            logger.info(f"LLM response parsed successfully: {response_dict}")
            return response_dict
        else:
            raise ValueError("No JSON found in LLM response")
            
    except Exception as e:
        logger.error(f"Failed to make the LLM request. Error: {e}")
        raise

def LLM_Validate_Output(user_question, notebook_output):
    """
    Ask the LLM to validate if the notebook output sufficiently answers the user's question.
    """
    logger.info("Validating notebook output with LLM")
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that evaluates if the notebook output sufficiently answers the user's question. "
                "The output doesn't need to be perfect, just functional and correct. "
                "If the output produces a valid result, even if it could be formatted better, consider it valid."
            )
        },
        {
            "role": "user",
            "content": (
                f"User's question: {user_question}\n"
                f"Notebook output:\n{notebook_output}\n"
                "Based on the notebook output, does it sufficiently answer the user's question? "
                "Answer with just 'Yes' or 'No' followed by a brief explanation."
            )
        }
    ]

    try:
        # response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=messages,
        #     temperature=0.0
        # )
        response = client.chat(
                model=MODEL,
                messages=messages,
                options={'temperature': 0.0}
            )
        answer = response.choices[0].message.content.strip()
        logger.debug(f"LLM validation answer: {answer}")

        # Parse the answer to determine yes/no
        yes_no_match = re.match(r'\s*(Yes|No)', answer, re.IGNORECASE)
        if yes_no_match:
            decision = yes_no_match.group(1).strip().lower()
            return decision == 'yes', answer
        else:
            # If we can't determine yes/no, assume it's valid to prevent endless loops
            logger.warning("Could not determine validation result, assuming valid.")
            return True, "Validation result unclear, proceeding with current output."

    except Exception as e:
        logger.error(f"Failed to make the validation request. Error: {e}")
        # In case of validation error, assume the output is valid
        return True, f"Validation error: {str(e)}"

def LLM_Answer(user_question, notebook_output):
    """
    Ask the LLM to answer the user's original question based on the notebook output.

    Parameters:
    - user_question: The original user's question.
    - notebook_output: The output from the executed notebook cells.

    Returns:
    - A JSON with the user's question, the answer, and sources.
    """
    logger.info("Generating final answer with LLM")
    
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that answers user questions based on the output of a Jupyter notebook."
        },
        {
            "role": "user",
            "content": (
                f"User's question: {user_question}\n"
                f"Notebook output:\n{notebook_output}\n"
                "Please answer the user's question based on the notebook output."
            )
        }
    ]

    try:
        # response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=messages,
        #     temperature=0.3
        # )
        response = client.chat(
                model=MODEL,
                messages=messages,
                options={'temperature': 0.3}
            )
        
        answer = response.choices[0].message.content
        return {
            "question": user_question,
            "answer": answer,
            "sources": notebook_output
        }
        
    except Exception as e:
        logger.error(f"Failed to make the answer request. Error: {e}")
        raise

def create_prompt(user_message, notebook_state, file_path=None, feedback=None):
    """Helper function to create the prompt for LLM requests."""
    if notebook_state.strip() == '':
        notebook_state_dict = {'cells': []}
    else:
        notebook_state_dict = json.loads(notebook_state)

    notebook_overview = "\n".join(
        [f"Cell {idx}:\n{cell['source']}" for idx, cell in enumerate(notebook_state_dict['cells'])]
    )

    prompt = f"""User's message: {user_message}
Notebook state:
{notebook_overview}
Number of cells in the notebook: {len(notebook_state_dict['cells'])}
"""

    if feedback:
        prompt += f"\nPrevious feedback: {feedback}\n"
    if file_path:
        prompt += f"\nData file location: {file_path}\n"

    return prompt

def code_requires_confirmation(code):
    """
    Analyzes the code to determine if it involves adding/deleting files or folders,
    or installing libraries (e.g., pip install).

    Returns True if confirmation is needed, False otherwise.
    """
    patterns = [
        r'\bpip\s+install\b',
        r'\bos\.remove\b',
        r'\bos\.rmdir\b',
        r'\bshutil\.rmtree\b',
        r'\bos\.mkdir\b',
        r'\bos\.makedirs\b',
        r'\bos\.unlink\b',
        r'\bopen\s*\(.*,\s*[\'"]w[\'"]\s*\)',
        r'\bopen\s*\(.*,\s*[\'"]wb[\'"]\s*\)',
        r'\bos\.system\b',
        r'\bsubprocess\.run\b',
        r'\bsubprocess\.Popen\b',
        r'\bimport\s+os\b.*\nos\.',
        r'\bimport\s+shutil\b.*\nshutil\.'
    ]

    for pattern in patterns:
        if re.search(pattern, code, re.MULTILINE):
            logger.warning("Code requires confirmation due to potentially dangerous operations.")
            return True
    return False

def clear_notebook_state(notebook_filename, executed_cells_filename):
    """
    Clears the notebook state and resets the executed cells.
    """
    try:
        logger.info("Clearing notebook state")
        # Clear the notebook file
        nb = nbf.v4.new_notebook()
        with open(notebook_filename, 'w') as f:
            nbf.write(nb, f)
        logger.info("Notebook cleared")

        # Clear executed_cells.json by writing an empty list
        with open(executed_cells_filename, 'w') as f:
            json.dump([], f)
        logger.info("Executed cells state cleared")
    except Exception as e:
        # Log the exception for debugging
        logger.error(f"Failed to reset the notebook state: {e}")
        raise Exception("Failed to reset the notebook state.")

def main(user_message, session, file_path, allow_dangerous_code=False):
    """
    Main function to process user messages and manage notebook operations.
    """
    if user_message.lower() == 'clear':
        clear_notebook_state(session.notebook_filename, session.executed_cells_filename)
        return {"detail": "Notebook state has been reset."}

    max_attempts = 10
    attempts = 0
    success = False
    feedback = None
    last_output = None

    while attempts < max_attempts and not success:
        attempts += 1
        logger.info(f"Attempt {attempts} of {max_attempts}")

        notebook_state_json = session.get_notebook()
        response_dict = LLM_Request(user_message, notebook_state_json, file_path, feedback=feedback)
        
        action = response_dict.get("action")
        cell_index = response_dict.get("cell_index")
        code = response_dict.get("code")
        needs_confirmation = response_dict.get("needs_confirmation", False)

        if code_requires_confirmation(code) and not allow_dangerous_code:
            raise ValueError("The generated code involves dangerous operations. Set 'allow_dangerous_code' to True to proceed.")

        try:
            if action == "add":
                result = session.add_cell(code)
                cell_index = len(session.nb['cells']) - 1
            elif action == "edit":
                if cell_index is None or cell_index >= len(session.nb['cells']) or cell_index < 0:
                    result = session.add_cell(code)
                    cell_index = len(session.nb['cells']) - 1
                else:
                    result = session.edit_cell(cell_index, code)
            elif action == "remove":
                if cell_index is not None and 0 <= cell_index < len(session.nb['cells']):
                    result = session.remove_cell(cell_index)
                else:
                    raise ValueError("Invalid cell index provided for removal.")
            else:
                raise ValueError("Invalid action provided by the assistant.")

            session.write_notebook_to_file()
            
            last_output, last_error = session.run_cell(cell_index)
            
            if last_error:
                feedback = last_error
                continue

            is_valid, validation_feedback = LLM_Validate_Output(user_message, last_output)
            if is_valid:
                success = True
                # Generate the final answer using the notebook output
                return LLM_Answer(user_message, last_output)
            else:
                feedback = validation_feedback
                if attempts == max_attempts:
                    # On last attempt, use the output anyway
                    return LLM_Answer(user_message, last_output)

        except Exception as e:
            feedback = str(e)
            logger.error(f"Error during execution: {feedback}")
            if attempts == max_attempts:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error during execution: {feedback}"
                )
            continue

    raise HTTPException(
        status_code=500,
        detail="Failed to generate a satisfactory response after maximum attempts"
    )

@app.post("/ask")
def ask_question(user_question: UserQuestion):
    try:
        logger.info(f"Received question: {user_question.question}")
        
        # Initialize the session per request
        notebook_filename = user_question.notebook_filename
        executed_cells_filename = 'executed_cells.json'
        session = NotebookSession(notebook_filename, executed_cells_filename)

        response_json = main(
            user_message=user_question.question,
            session=session,
            file_path=user_question.file_path,
            allow_dangerous_code=user_question.allow_dangerous_code
        )
        if response_json:
            if 'detail' in response_json:
                return JSONResponse(content=response_json, status_code=200)
            else:
                # Return the entire response, including the answer and sources
                logger.info("Returning final answer to the client")
                return response_json
        else:
            logger.error("Failed to process the question.")
            raise HTTPException(status_code=500, detail="Failed to process the question.")
    except Exception as e:
        # Log the exception and traceback for debugging
        traceback_str = traceback.format_exc()
        logger.error(f"Exception occurred: {e}\n{traceback_str}")
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)