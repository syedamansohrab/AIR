import fitz  # PyMuPDF
import os
import io
from PIL import Image

# Setup directories
CORPUS_DIR = "../data-corpus"
IMAGE_DIR = "extracted_images"

# Create the images folder if it doesn't exist
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def extract_images_from_pdfs():
    print("🚀 Starting Patent Image Extraction...")
    
    # Loop through every PDF in the corpus
    for filename in os.listdir(CORPUS_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(CORPUS_DIR, filename)
            pdf_document = fitz.open(pdf_path)
            
            image_count = 0
            
            # Go through every page
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                image_list = page.get_images(full=True)
                
                # Extract each image on the page
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    try:
                        # Save it with the patent name attached so we know where it came from!
                        image = Image.open(io.BytesIO(image_bytes))
                        clean_patent_name = filename.replace(".pdf", "")
                        image_filename = f"{clean_patent_name}_page{page_num+1}_img{img_index+1}.{image_ext}"
                        image_path = os.path.join(IMAGE_DIR, image_filename)
                        
                        image.save(open(image_path, "wb"))
                        image_count += 1
                    except Exception as e:
                        continue
                        
            print(f"✅ Extracted {image_count} diagrams from {filename}")

if __name__ == "__main__":
    extract_images_from_pdfs()
    print("\n🎉 All images extracted successfully! Ready for AI processing.")