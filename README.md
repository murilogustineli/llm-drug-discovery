# LLMs in Drug Discovery and Development - Streamlit Application

This document provides information on how the Streamlit application works and instructions on how to run it.

## Application Overview

This Streamlit application provides an interactive exploration of how Large Language Models (LLMs) are transforming the drug discovery and development process. Based on the research paper "Large Language Models in Drug Discovery and Development: From Disease Mechanisms to Clinical Trials", the application illustrates the impact of LLMs across the three main stages of drug development:

1. **Understanding Disease Mechanisms**
2. **Drug Discovery**
3. **Clinical Trials**

The application also explores the two main paradigms of LLMs (specialized and general-purpose), assesses the maturity of different LLM applications in the field, and discusses future directions.

## Application Structure

The application is organized into several pages accessible through the sidebar navigation:

1. **Home**: Provides an overview of the application and the drug discovery pipeline
2. **LLM Paradigms**: Explains the two main paradigms of language models used in drug discovery
3. **Disease Mechanisms**: Details how LLMs help in understanding disease mechanisms
4. **Drug Discovery**: Explores LLM applications in the drug discovery process
5. **Clinical Trials**: Describes how LLMs optimize clinical trial processes
6. **Maturity Assessment**: Visualizes the current state of LLM applications in the field
7. **Future Directions**: Discusses upcoming trends and challenges

## Interactive Features

The application includes several interactive elements:

- **Navigation Sidebar**: Allows users to move between different sections
- **Interactive Dropdowns**: Users can select specific applications to explore in detail
- **Comparative Views**: Toggle between different aspects of LLM paradigms
- **Filterable Visualizations**: Users can filter the maturity assessment by stage and maturity level
- **Tabbed Interfaces**: Organize complex information into accessible formats

## How to Run the Application

### Prerequisites

Before running the application, ensure you have Python installed on your system. This application was developed with Python 3.8+ in mind.

### Installation Steps

1. **Clone or download the application files** to your local machine

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

6. **Access the application** in your web browser. Streamlit will automatically open a browser window, or you can navigate to the URL displayed in the terminal (typically http://localhost:8501).

### Troubleshooting

- If you encounter any issues with dependencies, ensure you're using a compatible Python version and that all packages in requirements.txt are installed correctly.
- If the application doesn't open automatically, try accessing it manually by entering the URL shown in the terminal.
- If you see visualization errors, make sure Plotly and its dependencies are correctly installed.

## Customization

You can customize the application by:

- Adding new content to the existing pages
- Creating additional pages for more specific topics
- Enhancing visualizations with more data from the paper
- Incorporating additional research papers or resources

## Technical Details

The application is built using:

- **Streamlit**: For the web application framework
- **Pandas**: For data handling and manipulation
- **Plotly**: For interactive visualizations
- **Matplotlib**: For additional plotting capabilities
- **PIL (Pillow)**: For image processing

The code is structured with a main app.py file that contains all the page definitions and interactive elements.

## Paper Reference

This application is based on the paper:
"Large Language Models in Drug Discovery and Development: From Disease Mechanisms to Clinical Trials"
Available at: https://arxiv.org/pdf/2409.04481
