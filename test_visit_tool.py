import sys
from inference.tool_visit import Visit

def main():
    # We need to add the root of the project to the path, so the imports work
    # This is normally handled by running the script from the root
    # but we are doing it here just in case.
    sys.path.insert(0, ".")
    print("Testing the Visit tool...")
    visit_tool = Visit()
    params = {
        "url": "http://iamai.no",
        "goal": "Test visit"
    }
    try:
        result = visit_tool.call(params)
        print("\n--- Result ---")
        print(result)
        print("--- End Result ---\n")
        if "could not be accessed" in result:
            print("Test Failed: The tool could not access the webpage.")
        else:
            print("Test Succeeded: The tool successfully accessed the webpage.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Test Failed: The tool raised an exception.")


if __name__ == "__main__":
    main()
