import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import base64

# Set page configuration
st.set_page_config(
    page_title="LLMs in Drug Discovery",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS
def local_css():
    st.markdown(
        """
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #0D47A1;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #1565C0;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }
        .highlight {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .card {
            background-color: #FFFFFF;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #757575;
            font-size: 0.8rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


local_css()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "LLM Paradigms",
        "Disease Mechanisms",
        "Drug Discovery",
        "Clinical Trials",
        "Maturity Assessment",
        "Future Directions",
    ],
)

# Home page
if page == "Home":
    st.markdown(
        "<h1 class='main-header'>Large Language Models in Drug Discovery and Development</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align: center;'>From Disease Mechanisms to Clinical Trials</h3>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    This application provides an interactive exploration of how Large Language Models (LLMs) are transforming the drug discovery and development process. Based on the research paper ["Large Language Models in Drug Discovery and Development: From Disease Mechanisms to Clinical Trials"](https://arxiv.org/abs/2409.04481), this application illustrates the impact of LLMs across the three main stages of drug development.
    
    The integration of LLMs into drug discovery marks a significant paradigm shift, offering novel methodologies for understanding disease mechanisms, facilitating drug discovery, and optimizing clinical trial processes.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Drug Discovery Pipeline Visualization
    st.markdown(
        "<h2 class='sub-header'>Drug Discovery Pipeline</h2>", unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 class='section-header'>Understanding Disease Mechanisms</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        - Literature and patent analysis
        - Functional genomics analysis
        - Target gene identification
        - Biomarker analysis
        - Gene network analysis
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 class='section-header'>Drug Discovery</h3>", unsafe_allow_html=True
        )
        st.markdown("""
        - Virtual screening
        - Structure-based drug design
        - De novo molecule generation
        - Retrosynthetic planning
        - Reaction prediction
        - Protein-ligand binding prediction
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 class='section-header'>Clinical Trials</h3>", unsafe_allow_html=True
        )
        st.markdown("""
        - Patient-trial matching
        - Trial design automation
        - Trial result prediction
        - Trial result collection
        - Regulatory compliance
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("""
    ### Key Insights from the Paper
    
    - LLMs are revolutionizing the drug discovery pipeline, with applications across all three stages
    - The integration of LLMs can significantly reduce the time and resources required for drug development
    - Future drug discovery may include highly automated LLM applications across all stages
    - LLMs can help in understanding disease mechanisms, designing drug molecules, and optimizing clinical trials
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<p class='footer'>Based on the paper: Large Language Models in Drug Discovery and Development: From Disease Mechanisms to Clinical Trials</p>",
        unsafe_allow_html=True,
    )

# LLM Paradigms page
elif page == "LLM Paradigms":
    st.markdown(
        "<h1 class='main-header'>LLM Paradigms in Drug Discovery</h1>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    The paper identifies two main paradigms of language models used in drug discovery and development:
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 class='section-header'>Specialized Language Models</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        - Trained on specific scientific languages (chemistry, biology, proteins)
        - Tailored for specific science-related tasks
        - Used as tools to perform specific tasks
        - Users provide required information, model outputs prediction
        - Examples: Molecule SMILES, IUPAC, FASTA sequences
        - Applications: Retrosynthetic planning, reaction prediction, protein structure prediction
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 class='section-header'>General-Purpose Language Models</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        - Trained on diverse textual information from various sources
        - Can perform a wide range of tasks through natural language interaction
        - Used like an assistant that users interact with using plain language
        - Capabilities: Prior knowledge understanding, reasoning, role playing, planning
        - Sources: Books, internet, papers, social media
        - Applications: Information retrieval, explanation, guidance
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<h2 class='sub-header'>Comparison of LLM Paradigms</h2>",
        unsafe_allow_html=True,
    )

    # Interactive comparison
    comparison_option = st.selectbox(
        "Select aspect to compare:",
        [
            "Training Data",
            "Usage Pattern",
            "Capabilities",
            "Applications in Drug Discovery",
        ],
    )

    if comparison_option == "Training Data":
        data = {
            "Specialized LLMs": [
                "Scientific languages",
                "Domain-specific corpora",
                "Structured data (SMILES, FASTA)",
                "Curated datasets",
            ],
            "General-Purpose LLMs": [
                "Diverse text sources",
                "Books and articles",
                "Internet content",
                "Scientific papers",
            ],
        }
    elif comparison_option == "Usage Pattern":
        data = {
            "Specialized LLMs": [
                "Tool-like usage",
                "Specific input format required",
                "Focused on single tasks",
                "Requires domain expertise",
            ],
            "General-Purpose LLMs": [
                "Assistant-like usage",
                "Natural language interaction",
                "Conversational",
                "Accessible to non-experts",
            ],
        }
    elif comparison_option == "Capabilities":
        data = {
            "Specialized LLMs": [
                "High accuracy in specific tasks",
                "Limited to trained domain",
                "Structured outputs",
                "Optimized performance",
            ],
            "General-Purpose LLMs": [
                "Broad knowledge base",
                "Reasoning across domains",
                "Natural language generation",
                "Adaptability",
            ],
        }
    else:  # Applications
        data = {
            "Specialized LLMs": [
                "Molecule generation",
                "Protein structure prediction",
                "Binding affinity prediction",
                "Reaction prediction",
            ],
            "General-Purpose LLMs": [
                "Literature review",
                "Hypothesis generation",
                "Experimental design",
                "Knowledge synthesis",
            ],
        }

    df = pd.DataFrame(data)
    st.table(df)

    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("""
    ### Key Takeaway
    
    Both paradigms of LLMs have important roles in drug discovery and development. Specialized models excel at specific scientific tasks with high accuracy, while general-purpose models provide broader context, reasoning, and natural language interaction. The most effective approaches often combine both types of models.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Disease Mechanisms page
elif page == "Disease Mechanisms":
    st.markdown(
        "<h1 class='main-header'>Understanding Disease Mechanisms with LLMs</h1>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    The first stage in drug discovery and development is understanding disease mechanisms. LLMs can significantly enhance this process through various applications:
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Applications in disease mechanism understanding
    applications = [
        "Literature Review and Patent Analysis",
        "Functional Genomics Analysis",
        "Target Gene Identification",
        "Biomarker Analysis",
        "Gene Network Analysis",
    ]

    selected_application = st.selectbox(
        "Select an application to explore:", applications
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if selected_application == "Literature Review and Patent Analysis":
        st.markdown(
            "<h3 class='section-header'>Literature Review and Patent Analysis</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can perform comprehensive literature reviews and patent analyses to explore biological pathways involved in diseases:
        
        - **Automated extraction** of relevant information from thousands of research papers
        - **Identification of patterns** across multiple studies that might be missed by human researchers
        - **Summarization** of complex scientific findings into accessible formats
        - **Connection of disparate research** across different fields
        - **Tracking of patent landscapes** to identify promising research directions
        
        This capability allows researchers to quickly gain insights from vast amounts of scientific literature, accelerating the understanding of disease mechanisms.
        """)

    elif selected_application == "Functional Genomics Analysis":
        st.markdown(
            "<h3 class='section-header'>Functional Genomics Analysis</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can assist in functional genomics analysis to understand gene functions and their roles in diseases:
        
        - **Analysis of gene expression data** to identify patterns related to disease states
        - **Interpretation of genomic variants** and their potential impact on protein function
        - **Prediction of gene function** based on sequence and structural information
        - **Integration of multi-omics data** for comprehensive understanding
        - **Identification of genetic pathways** involved in disease progression
        
        These capabilities help researchers understand the genetic basis of diseases, which is crucial for identifying potential drug targets.
        """)

    elif selected_application == "Target Gene Identification":
        st.markdown(
            "<h3 class='section-header'>Target Gene Identification</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can help identify target genes that may be suitable for therapeutic intervention:
        
        - **Prioritization of genes** based on their association with disease
        - **Assessment of druggability** of potential target proteins
        - **Prediction of off-target effects** to minimize side effects
        - **Identification of novel targets** that may have been overlooked
        - **Evaluation of target validation evidence** from experimental data
        
        By analyzing gene-related literature and experimental results, LLMs can compare data on various genes and recommend those with favorable characteristics for drug development.
        """)

    elif selected_application == "Biomarker Analysis":
        st.markdown(
            "<h3 class='section-header'>Biomarker Analysis</h3>", unsafe_allow_html=True
        )
        st.markdown("""
        LLMs can assist in identifying and validating biomarkers for diseases:
        
        - **Discovery of potential biomarkers** from literature and experimental data
        - **Validation of biomarker utility** across different studies
        - **Correlation of biomarkers** with disease progression and treatment response
        - **Identification of biomarker panels** for improved diagnostic accuracy
        - **Prediction of biomarker performance** in different patient populations
        
        Biomarkers are crucial for patient stratification, treatment monitoring, and as surrogate endpoints in clinical trials.
        """)

    else:  # Gene Network Analysis
        st.markdown(
            "<h3 class='section-header'>Gene Network Analysis</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can analyze complex gene networks to understand disease mechanisms:
        
        - **Construction of gene interaction networks** from literature and databases
        - **Identification of key regulatory nodes** within networks
        - **Prediction of network perturbations** caused by genetic variants or drugs
        - **Discovery of disease modules** within larger biological networks
        - **Integration of network information** with other biological data
        
        Understanding gene networks provides insights into the complex interplay of genes in disease states and helps identify potential points of intervention.
        """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("""
    ### Case Study: Geneformer
    
    The paper mentions Geneformer, an LLM pretrained on 30 million single-cell transcriptomes, which has successfully identified candidate therapeutic targets for cardiomyopathy via in silico deletion. This demonstrates how specialized LLMs can directly contribute to target identification for complex diseases.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Drug Discovery page
elif page == "Drug Discovery":
    st.markdown(
        "<h1 class='main-header'>Drug Discovery with LLMs</h1>", unsafe_allow_html=True
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    The second stage in drug development is drug discovery, where potential therapeutic compounds are identified and optimized. LLMs are revolutionizing this process through various applications:
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Applications in drug discovery
    applications = [
        "Virtual Screening",
        "Structure-based Drug Design",
        "De Novo Molecule Generation",
        "Retrosynthetic Planning",
        "Reaction Prediction",
        "Protein-Ligand Binding Prediction",
    ]

    selected_application = st.selectbox(
        "Select an application to explore:", applications
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if selected_application == "Virtual Screening":
        st.markdown(
            "<h3 class='section-header'>Virtual Screening</h3>", unsafe_allow_html=True
        )
        st.markdown("""
        LLMs can enhance virtual screening processes to identify promising drug candidates:
        
        - **Filtering large compound libraries** to identify molecules with desired properties
        - **Predicting binding affinities** between compounds and target proteins
        - **Identifying novel chemical scaffolds** with potential activity against targets
        - **Prioritizing compounds** for experimental validation
        - **Incorporating multiple parameters** for multi-objective optimization
        
        Virtual screening with LLMs can significantly reduce the number of compounds that need to be tested experimentally, saving time and resources.
        """)

    elif selected_application == "Structure-based Drug Design":
        st.markdown(
            "<h3 class='section-header'>Structure-based Drug Design</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can assist in structure-based drug design by leveraging protein structural information:
        
        - **Analysis of protein binding sites** to identify key interaction points
        - **Prediction of protein-ligand interactions** to optimize binding
        - **Suggestion of chemical modifications** to improve drug properties
        - **Evaluation of conformational changes** upon ligand binding
        - **Integration of structural knowledge** with other data sources
        
        By understanding the structural basis of protein-drug interactions, LLMs can guide the design of more effective and selective compounds.
        """)

    elif selected_application == "De Novo Molecule Generation":
        st.markdown(
            "<h3 class='section-header'>De Novo Molecule Generation</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can generate novel molecular structures with desired properties:
        
        - **Creation of new chemical entities** not present in existing databases
        - **Optimization of molecular properties** such as solubility, permeability, and stability
        - **Generation of molecules** similar to known active compounds but with improved properties
        - **Design of molecules** that target specific protein binding sites
        - **Interactive refinement** of generated molecules based on expert feedback
        
        De novo molecule generation with LLMs offers an interactive platform aiding experts in discovering novel and effective compounds through suggestions for molecule editing and generation.
        """)

    elif selected_application == "Retrosynthetic Planning":
        st.markdown(
            "<h3 class='section-header'>Retrosynthetic Planning</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can plan synthetic routes for complex molecules:
        
        - **Breaking down target molecules** into simpler precursors
        - **Suggesting viable synthetic pathways** based on known chemical reactions
        - **Evaluating synthetic accessibility** of proposed compounds
        - **Optimizing synthetic routes** for efficiency and yield
        - **Considering available reagents and conditions** for practical implementation
        
        Retrosynthetic planning with LLMs helps bridge the gap between computational drug design and experimental synthesis.
        """)

    elif selected_application == "Reaction Prediction":
        st.markdown(
            "<h3 class='section-header'>Reaction Prediction</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can predict the outcomes of chemical reactions:
        
        - **Forecasting reaction products** from given reactants and conditions
        - **Estimating reaction yields** and selectivity
        - **Identifying potential side reactions** and byproducts
        - **Suggesting optimal reaction conditions** for desired transformations
        - **Predicting stereochemical outcomes** of reactions
        
        Reaction prediction capabilities help chemists plan and optimize synthetic procedures for drug candidates.
        """)

    else:  # Protein-Ligand Binding Prediction
        st.markdown(
            "<h3 class='section-header'>Protein-Ligand Binding Prediction</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can predict how drug candidates interact with target proteins:
        
        - **Estimating binding affinities** between compounds and proteins
        - **Identifying key interaction points** in protein-ligand complexes
        - **Predicting binding poses** of ligands in protein binding sites
        - **Evaluating selectivity** across related protein targets
        - **Assessing the impact of protein mutations** on binding
        
        Accurate prediction of protein-ligand interactions is crucial for understanding drug efficacy and specificity.
        """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("""
    ### Case Studies from the Paper
    
    The paper mentions several LLMs that have shown promise in drug discovery:
    
    - **Chemcrow**: An LLM that has demonstrated potential in automating chemistry experiments related to drug discovery, specifically in directed synthesis and chemical reaction prediction.
    
    - **LLM4SD**: Showed that LLMs can perform scientific synthesis, inference, and explanation directly from raw experimental data and formulate hypotheses that resonate with human experts' analysis.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Clinical Trials page
elif page == "Clinical Trials":
    st.markdown(
        "<h1 class='main-header'>Optimizing Clinical Trials with LLMs</h1>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    The third stage in drug development involves clinical trials to test the safety and efficacy of potential treatments. LLMs can help optimize various aspects of this process:
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Applications in clinical trials
    applications = [
        "Patient-Trial Matching",
        "Trial Design Automation",
        "Trial Result Prediction",
        "Trial Result Collection",
        "Regulatory Compliance",
    ]

    selected_application = st.selectbox(
        "Select an application to explore:", applications
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if selected_application == "Patient-Trial Matching":
        st.markdown(
            "<h3 class='section-header'>Patient-Trial Matching</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can streamline the process of matching patients with appropriate clinical trials:
        
        - **Analysis of patient records** to identify eligible candidates
        - **Interpretation of complex inclusion/exclusion criteria**
        - **Matching patient characteristics** with trial requirements
        - **Prioritizing trials** based on patient needs and preferences
        - **Identifying potential barriers** to patient participation
        
        Efficient patient-trial matching can accelerate recruitment, reduce costs, and ensure that trials include appropriate participant populations.
        """)

    elif selected_application == "Trial Design Automation":
        st.markdown(
            "<h3 class='section-header'>Trial Design Automation</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can assist in designing more effective clinical trials:
        
        - **Optimizing trial protocols** based on historical data
        - **Suggesting appropriate endpoints** and outcome measures
        - **Determining optimal sample sizes** for statistical power
        - **Designing adaptive trial strategies** that evolve based on interim results
        - **Identifying potential confounding factors** to control for
        
        Automated trial design can lead to more efficient studies that require fewer resources while maintaining scientific rigor.
        """)

    elif selected_application == "Trial Result Prediction":
        st.markdown(
            "<h3 class='section-header'>Trial Result Prediction</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can predict potential outcomes of clinical trials:
        
        - **Forecasting trial success probability** based on drug characteristics and trial design
        - **Identifying patient subgroups** most likely to benefit from treatment
        - **Predicting potential adverse events** based on drug properties
        - **Estimating effect sizes** for primary and secondary endpoints
        - **Simulating trial outcomes** under different scenarios
        
        Predictive capabilities can help prioritize drug candidates and optimize resource allocation in drug development.
        """)

    elif selected_application == "Trial Result Collection":
        st.markdown(
            "<h3 class='section-header'>Trial Result Collection</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can enhance the collection and analysis of clinical trial data:
        
        - **Extracting relevant information** from clinical notes and reports
        - **Standardizing data** from different sources and formats
        - **Identifying data quality issues** and inconsistencies
        - **Summarizing complex trial results** for different stakeholders
        - **Integrating results** across multiple trials for meta-analysis
        
        Improved data collection and analysis can lead to more robust conclusions and better-informed decision-making.
        """)

    else:  # Regulatory Compliance
        st.markdown(
            "<h3 class='section-header'>Regulatory Compliance</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        LLMs can assist in navigating complex regulatory requirements:
        
        - **Interpreting regulatory guidelines** from different agencies
        - **Ensuring protocol compliance** with regulatory standards
        - **Generating regulatory documentation** based on trial data
        - **Identifying potential regulatory concerns** early in development
        - **Tracking regulatory changes** and their implications
        
        Regulatory compliance support can help avoid delays and ensure that submissions meet all requirements.
        """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("""
    ### Case Study: Med-PaLM
    
    The paper mentions Med-PaLM, a large language model encoding clinical knowledge, which was the first to reach human expert level in USMLE-styled questions. This advancement highlights the potential of LLMs to liberate clinical practitioners from laborious activities associated with clinical trials and to provide expert-level medical knowledge support.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Maturity Assessment page
elif page == "Maturity Assessment":
    st.markdown(
        "<h1 class='main-header'>Maturity Assessment of LLM Applications</h1>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    The paper evaluates the current state of LLM applications in drug discovery and development, classifying each into one of four categories:
    
    - **Not Applicable**: Areas where LLMs are not currently applicable
    - **Nascent**: Early-stage applications with proof-of-concept demonstrations
    - **Advanced**: Applications with significant development and some real-world testing
    - **Mature**: Well-established applications with demonstrated value in real-world settings
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Create maturity data
    maturity_data = {
        "Application": [
            "Literature Review",
            "Target Identification",
            "Biomarker Analysis",
            "Gene Network Analysis",
            "Virtual Screening",
            "De Novo Molecule Generation",
            "Retrosynthetic Planning",
            "Reaction Prediction",
            "Protein-Ligand Binding",
            "Patient-Trial Matching",
            "Trial Design",
            "Trial Result Prediction",
        ],
        "Stage": [
            "Disease Mechanisms",
            "Disease Mechanisms",
            "Disease Mechanisms",
            "Disease Mechanisms",
            "Drug Discovery",
            "Drug Discovery",
            "Drug Discovery",
            "Drug Discovery",
            "Drug Discovery",
            "Clinical Trials",
            "Clinical Trials",
            "Clinical Trials",
        ],
        "Maturity": [
            "Mature",
            "Advanced",
            "Advanced",
            "Advanced",
            "Advanced",
            "Advanced",
            "Advanced",
            "Advanced",
            "Nascent",
            "Nascent",
            "Nascent",
            "Nascent",
        ],
    }

    df = pd.DataFrame(maturity_data)

    # Create a categorical color map
    color_map = {
        "Not Applicable": "#E0E0E0",
        "Nascent": "#FFECB3",
        "Advanced": "#90CAF9",
        "Mature": "#A5D6A7",
    }

    # Filter options
    stage_filter = st.multiselect(
        "Filter by Stage:",
        options=["Disease Mechanisms", "Drug Discovery", "Clinical Trials"],
        default=["Disease Mechanisms", "Drug Discovery", "Clinical Trials"],
    )

    maturity_filter = st.multiselect(
        "Filter by Maturity Level:",
        options=["Not Applicable", "Nascent", "Advanced", "Mature"],
        default=["Nascent", "Advanced", "Mature"],
    )

    # Apply filters
    filtered_df = df[
        df["Stage"].isin(stage_filter) & df["Maturity"].isin(maturity_filter)
    ]

    # Create visualization
    if not filtered_df.empty:
        fig = px.bar(
            filtered_df,
            x="Application",
            y="Maturity",
            color="Maturity",
            color_discrete_map=color_map,
            category_orders={
                "Maturity": ["Not Applicable", "Nascent", "Advanced", "Mature"]
            },
            labels={"Application": "Application Area", "Maturity": "Maturity Level"},
            title="Maturity Assessment of LLM Applications in Drug Discovery and Development",
            height=500,
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            legend_title_text="Maturity Level",
            plot_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available with the selected filters.")

    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("""
    ### Key Observations
    
    - **Literature Review** is the most mature application of LLMs in drug discovery, leveraging their natural language processing capabilities
    - Applications in the **Disease Mechanisms** stage are generally more advanced than those in later stages
    - **Clinical Trial** applications are mostly in the nascent stage, indicating significant room for growth
    - **Drug Discovery** applications show varying levels of maturity, with some areas like virtual screening more advanced than others
    
    This assessment provides an overview of the current state in the field and indicates promising future directions for research and development.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Future Directions page
elif page == "Future Directions":
    st.markdown(
        "<h1 class='main-header'>Future Directions of LLMs in Drug Discovery</h1>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    The paper discusses several important future directions for the development and application of LLMs in drug discovery and development:
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Future directions tabs
    tabs = st.tabs(
        [
            "Technical Advancements",
            "Biological Applications",
            "Ethical Considerations",
            "Integration Challenges",
        ]
    )

    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 class='section-header'>Technical Advancements</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        Several technical challenges need to be addressed to enhance LLM capabilities in drug discovery:
        
        - **Reducing Hallucinations**: Improving the factual accuracy of LLM outputs to ensure reliable scientific information
        
        - **Expanding Context Windows**: Developing methods to handle longer contexts to process comprehensive scientific literature and data
        
        - **Enhancing Interpretability**: Creating more transparent models that can explain their reasoning and predictions
        
        - **Improving Scientific Understanding**: Developing LLMs with deeper understanding of scientific principles and reasoning
        
        - **Multimodal Capabilities**: Integrating text, image, and structural data for comprehensive analysis
        
        - **Domain-Specific Pretraining**: Creating specialized models for different aspects of drug discovery
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 class='section-header'>Biological Applications</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        Future biological applications of LLMs in drug discovery may include:
        
        - **Automated Target Identification**: LLMs could automatically identify promising drug targets by analyzing genomic and literature data
        
        - **Discovery of Biochemical Principles**: LLMs might infer new insights and uncover principles of biochemistry and pharmacology
        
        - **Design of New Therapeutic Modalities**: Beyond small molecules, LLMs could assist in designing gene therapies, biologics, and other novel therapeutic approaches
        
        - **Automated Experimentation**: LLMs could control robotic equipment to design and execute experiments with minimal human intervention
        
        - **Interactive Drug Design**: Creating platforms where experts and LLMs collaborate to design optimal drug candidates
        
        - **Personalized Medicine Applications**: Using LLMs to match treatments to individual patient characteristics
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 class='section-header'>Ethical Considerations</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        Important ethical considerations for the future development of LLMs in drug discovery include:
        
        - **Privacy Concerns**: Ensuring patient data used for training or analysis is properly protected
        
        - **Fairness and Bias**: Addressing potential biases in training data that could lead to inequitable drug development
        
        - **Transparency**: Making the decision-making processes of LLMs transparent to users and regulators
        
        - **Accountability**: Establishing clear lines of responsibility for LLM-assisted decisions in drug development
        
        - **Access and Equity**: Ensuring that LLM technologies benefit diverse populations and are not limited to wealthy institutions
        
        - **Regulatory Compliance**: Developing frameworks for validating and regulating LLM applications in healthcare
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 class='section-header'>Integration Challenges</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        Challenges in integrating LLMs into existing drug discovery workflows include:
        
        - **Interoperability**: Ensuring LLMs can work with existing software and databases
        
        - **Validation Protocols**: Developing methods to validate LLM outputs in scientific contexts
        
        - **Training Requirements**: Educating scientists and researchers on effective use of LLM tools
        
        - **Computational Resources**: Addressing the substantial computing requirements of advanced LLMs
        
        - **Workflow Integration**: Seamlessly incorporating LLMs into established drug discovery processes
        
        - **Balancing Automation and Expertise**: Finding the optimal balance between LLM automation and human expertise
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("""
    ### Vision for the Future
    
    The paper envisions a future where LLMs enable a highly automated drug discovery process across all three stages. This could significantly reduce the time and resources required to bring new drugs to patients, potentially transforming the current 10-15 year, $2+ billion process into a more efficient and accessible endeavor.
    
    However, realizing this vision will require addressing technical challenges, ethical considerations, and integration issues while ensuring that human expertise remains central to the drug discovery process.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<p class='footer'>Application based on the paper: Large Language Models in Drug Discovery and Development: From Disease Mechanisms to Clinical Trials</p>",
    unsafe_allow_html=True,
)
