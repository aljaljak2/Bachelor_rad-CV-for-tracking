#!/usr/bin/env python3
"""
Script to generate a PDF document containing all Python code files
organized by folders with proper formatting and table of contents.
"""

import os
import glob
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfgen import canvas
from reportlab.platypus.tableofcontents import TableOfContents
import sys

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []
        
    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()
        
    def save(self):
        num_pages = len(self._saved_page_states)
        for (page_num, page_state) in enumerate(self._saved_page_states):
            self.__dict__.update(page_state)
            self.draw_page_number(page_num + 1, num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
        
    def draw_page_number(self, page_num, total_pages):
        self.setFont("Helvetica", 9)
        self.drawRightString(A4[0] - 0.75*inch, 0.75*inch, 
                           f"Page {page_num} of {total_pages}")

def collect_python_files():
    """Collect all Python files organized by directory."""
    base_path = Path(".")
    files_by_folder = {}
    
    # Get all Python files
    python_files = list(base_path.glob("**/*.py"))
    
    for file_path in python_files:
        # Get relative path from base
        rel_path = file_path.relative_to(base_path)
        folder = str(rel_path.parent)
        
        if folder == ".":
            folder = "Root Directory"
        
        if folder not in files_by_folder:
            files_by_folder[folder] = []
        
        files_by_folder[folder].append(file_path)
    
    return files_by_folder

def read_file_content(file_path):
    """Read file content with proper encoding handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def create_pdf_document():
    """Create the PDF document with all code files."""
    
    # Collect files
    files_by_folder = collect_python_files()
    
    # Setup PDF
    output_filename = "Tennis_Court_Tracking_Code_Documentation.pdf"
    doc = SimpleDocTemplate(output_filename, pagesize=A4, 
                          rightMargin=0.75*inch, leftMargin=0.75*inch,
                          topMargin=1*inch, bottomMargin=1*inch)
    
    # Setup styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    folder_style = ParagraphStyle(
        'FolderTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=30,
        textColor=colors.darkred,
        keepWithNext=1
    )
    
    file_style = ParagraphStyle(
        'FileTitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkgreen,
        keepWithNext=1
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontSize=9,
        spaceAfter=12,
        fontName='Courier-Bold',
        leftIndent=15,
        rightIndent=15,
        backgroundColor=colors.lightgrey,
        leading=11,
        wordWrap='CJK'
    )
    
    # Build content
    story = []
    
    # Title page
    story.append(Paragraph("Tennis Court Tracking System", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Complete Code Documentation", styles['Heading2']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 24))
    
    # Project overview
    overview_text = """
    This document contains all Python code files from the Tennis Court Tracking project.
    The project implements computer vision techniques for tennis court detection, player tracking,
    and distance measurement using YOLO object detection and DeepSORT tracking algorithms.
    
    The code is organized into the following main components:
    • Detection and Tracking modules for player and ball tracking
    • Distance Measurement modules for court mapping and coordinate transformation  
    • Supporting utilities and helper functions
    """
    story.append(Paragraph("Project Overview", styles['Heading2']))
    story.append(Paragraph(overview_text, styles['Normal']))
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    toc_data = [["Folder/File", "Page"]]
    current_page = 3  # Starting after title and TOC
    
    # Calculate approximate pages for TOC
    for folder, files in sorted(files_by_folder.items()):
        toc_data.append([f"[FOLDER] {folder}", str(current_page)])
        current_page += 1
        
        for file_path in sorted(files):
            filename = file_path.name
            content = read_file_content(file_path)
            # Rough estimate: 50 lines per page
            estimated_pages = max(1, len(content.split('\n')) // 50)
            toc_data.append([f"   {filename}", str(current_page)])
            current_page += estimated_pages
    
    toc_table = Table(toc_data, colWidths=[4*inch, 1*inch])
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(toc_table)
    story.append(PageBreak())
    
    # Add code content
    for folder, files in sorted(files_by_folder.items()):
        # Folder header
        story.append(Paragraph(f"[FOLDER] {folder}", folder_style))
        story.append(Spacer(1, 12))
        
        if folder != "Root Directory":
            story.append(Paragraph(f"Location: {folder}", styles['Italic']))
            story.append(Spacer(1, 12))
        
        # Files in folder
        for file_path in sorted(files):
            filename = file_path.name
            story.append(Paragraph(f"[FILE] {filename}", file_style))
            story.append(Spacer(1, 6))
            
            # File path
            story.append(Paragraph(f"Path: {file_path}", styles['Italic']))
            story.append(Spacer(1, 6))
            
            # File content
            content = read_file_content(file_path)
            
            if content:
                # Process content to preserve indentation
                lines = content.split('\n')
                processed_lines = []
                
                for line in lines:
                    # Convert tabs to 4 spaces for consistent indentation
                    line = line.expandtabs(4)
                    # Escape special characters for reportlab
                    line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    
                    # Handle very long lines by breaking them intelligently
                    max_line_length = 100
                    if len(line) > max_line_length:
                        # Try to break at logical points (spaces, commas, operators)
                        indent = len(line) - len(line.lstrip())
                        while len(line) > max_line_length:
                            break_point = max_line_length
                            # Look for good break points
                            for char in [' ', ',', '(', ')', '[', ']', '{', '}', '=', '+', '-']:
                                pos = line.rfind(char, 0, max_line_length)
                                if pos > max_line_length * 0.7:  # Don't break too early
                                    break_point = pos + 1
                                    break
                            
                            processed_lines.append(line[:break_point].rstrip())
                            line = ' ' * (indent + 4) + line[break_point:].lstrip()
                    
                    # Preserve leading spaces by converting them to non-breaking spaces
                    leading_spaces = len(line) - len(line.lstrip())
                    if leading_spaces > 0:
                        line = '&nbsp;' * leading_spaces + line.lstrip()
                    processed_lines.append(line)
                
                # Split content into chunks to avoid reportlab issues with very long paragraphs
                chunk_size = 50  # lines per chunk (reduced for better formatting)
                
                for i in range(0, len(processed_lines), chunk_size):
                    chunk_lines = processed_lines[i:i+chunk_size]
                    # Join lines with <br/> tags to preserve line breaks
                    chunk = '<br/>'.join(chunk_lines)
                    story.append(Paragraph(f"<font name='Courier' size='9'>{chunk}</font>", code_style))
            else:
                story.append(Paragraph("File is empty or could not be read.", styles['Italic']))
            
            story.append(Spacer(1, 20))
        
        story.append(PageBreak())
    
    # Build PDF
    print(f"Generating PDF: {output_filename}")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generated successfully: {output_filename}")
    
    return output_filename

if __name__ == "__main__":
    try:
        # Check if reportlab is available
        import reportlab
        output_file = create_pdf_document()
        print(f"\nSUCCESS! PDF created: {output_file}")
        print(f"Total Python files processed: {len(list(Path('.').glob('**/*.py')))}")
        
    except ImportError:
        print("ERROR: reportlab library is required to generate PDF.")
        print("Install it with: pip install reportlab")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR generating PDF: {str(e)}")
        sys.exit(1)