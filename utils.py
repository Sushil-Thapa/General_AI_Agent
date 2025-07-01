import time

def save_file_with_timestamp(content: str, file_name: str, extension: str) -> str:
    """
    Save content to a file with a timestamp.
    Args:
        content (str): The content to save.
        file_name (str): The base name of the file.
    Returns:
        str: The path to the saved file.
    """
    try:
        # save content to a file in test folder before returning
        # compute filepath with correct extension based on convert_to_markdown and add a timestamp for unicity
        
        unicity_suffix = str(int(time.time()))
        
        file_path = f"test/{file_name}_{unicity_suffix}.{extension}"
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"Error saving content to file: {e}")
    return file_name