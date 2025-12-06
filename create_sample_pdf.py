"""
Create a sample PDF for testing
Requires: pip install reportlab --break-system-packages
"""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path


def create_sample_pdf():
    """Create a simple test PDF"""
    
    # Ensure uploads directory exists
    Path("uploads").mkdir(exist_ok=True)
    
    output_path = "uploads/sample.pdf"
    
    # Create PDF
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Company Knowledge Base")
    
    c.setFont("Helvetica", 12)
    y_position = height - 150
    
    content = [
        "Introduction to Our Company",
        "",
        "Our company was founded in 2020 with a mission to revolutionize",
        "the way businesses handle their knowledge management systems.",
        "",
        "We specialize in artificial intelligence and machine learning",
        "solutions that help organizations make better decisions through",
        "data-driven insights.",
        "",
        "Our core values include:",
        "- Innovation and creativity",
        "- Customer-centric approach",
        "- Continuous improvement",
        "- Ethical AI development",
    ]
    
    for line in content:
        c.drawString(100, y_position, line)
        y_position -= 20
    
    c.showPage()
    
    # Page 2
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Our Products and Services")
    
    c.setFont("Helvetica", 12)
    y_position = height - 150
    
    content2 = [
        "We offer a range of AI-powered solutions:",
        "",
        "1. Smart Document Processing",
        "   Automatically extract and analyze information from documents",
        "   using advanced natural language processing.",
        "",
        "2. Intelligent Q&A Systems",
        "   Build knowledge bases that can answer questions naturally",
        "   and accurately using retrieval-augmented generation.",
        "",
        "3. Custom AI Solutions",
        "   Tailored artificial intelligence implementations for",
        "   specific business needs and challenges.",
    ]
    
    for line in content2:
        c.drawString(100, y_position, line)
        y_position -= 20
    
    c.save()
    
    print(f"✓ Sample PDF created: {output_path}")


if __name__ == "__main__":
    create_sample_pdf()