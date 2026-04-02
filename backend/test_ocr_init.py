from dots_ocr.parser import DotsOCRParser
try:
    print("Initialising DotsOCRParser...")
    parser = DotsOCRParser()
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
