import chromadb
import os
import openai
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from flask import Flask, render_template, request
import json
import logging
from typing import Dict, Any

# Load environment variables and set up OpenAI API key
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
openai_client = OpenAI()

# Set up paths for the Chroma database
chroma_path = 'chroma'
path = 'data'

# Initialize the Chroma client
chroma_client = chromadb.PersistentClient(path=chroma_path)


def setup_retrieval():
    """
    Set up the retrieval system using Chroma and LangChain.
    This function initializes the embedding model, vector store, and retriever.
    """
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        client=chroma_client,
        collection_name='pesticides_docs',
        embedding_function=embeddings
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Define a custom prompt template for the retrieval system
    prompt_template = """Use the following pieces of context to answer the question at the end.
    Provide detailed information about pesticide application procedures, safety considerations,
    and environmental impact. Include specific details about the pesticide if available, such
    as its chemical composition and recommended usage.

    Consider the following aspects in your answer:
    1. Application method and timing
    2. Safety precautions for farmers
    3. Environmental impact (soil, water, air)
    4. Effects on beneficial insects and wildlife
    5. Pre-harvest interval and chemical residue concerns
    6. Alternatives or integrated pest management strategies

    If the question is about a specific pesticide and you don't have information about it, suggest similar pesticides
    or general best practices.

    If you don't know the answer, say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}

    Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Set up the retrieval QA chain
    chain_type_kwargs = {"prompt": PROMPT}
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )


def check_chroma_collection():
    """
    Check if the Chroma collection exists and has documents.
    """
    try:
        collection = chroma_client.get_or_create_collection("pesticides_docs")
        count = collection.count()
        if count > 0:
            print(f"Chroma collection found with {count} documents")
            return collection
        else:
            print("Chroma collection exists but is empty")
            return None
    except Exception as e:
        print(e)
        return None


def load_model():
    """
    Initialize the retrieval model.
    """
    return setup_retrieval()


def get_model_response(query):
    """
    Get a response from the model for a given query.
    """
    response = model({"query": query})
    return response['result']


# Initialize Flask app
app = Flask(__name__)

# Load the model
model = load_model()


def get_structured_general_info(pesticide: str) -> Dict[str, Any]:
    query = f"""Provide a comprehensive yet easy-to-understand overview of {pesticide} in the following JSON format:
    {{
        "overview": "Detailed overview of the pesticide (400-500 words)",
        "classification": "Simple explanation of the chemical classification (50-100 words)",
        "primary_uses": [
            "Use 1 with detailed explanation (100-150 words)",
            "Use 2 with detailed explanation (100-150 words)",
            "Use 3 with detailed explanation (100-150 words)"
        ],
        "key_characteristics": [
            "Characteristic 1 with detailed explanation (100-150 words)",
            "Characteristic 2 with detailed explanation (100-150 words)",
            "Characteristic 3 with detailed explanation (100-150 words)"
        ],
        "toxicity_level": "Numerical value on a scale of 1-10",
        "efficacy_rating": "Numerical value on a scale of 1-10"
    }}
    Use simple language suitable for farmers. Avoid technical jargon and explain any complex terms.
    Provide practical, actionable information. If exact information is not available, provide the best estimate
    or state 'Unknown' for text fields and 0 for numerical fields.
    Ensure that the response is a valid JSON object."""

    try:
        raw_result = get_model_response(query)
        logging.info(f"Raw result for {pesticide}: {raw_result}")

        # Attempt to parse the JSON
        structured_data = json.loads(raw_result)

        # Validate and clean the structured data
        validated_data = {
            'overview': structured_data.get('overview', 'Information not available'),
            'classification': structured_data.get('classification', 'Unknown'),
            'primary_uses': structured_data.get('primary_uses', [])[:3],  # Limit to 3 uses
            'key_characteristics': structured_data.get('key_characteristics', [])[:3],  # Limit to 3 characteristics
            'toxicity_level': 0,
            'efficacy_rating': 0,
            'chart_data': {}
        }

        # Process toxicity and efficacy
        try:
            validated_data['toxicity_level'] = float(structured_data.get('toxicity_level', 0))
        except ValueError:
            validated_data['toxicity_level'] = 0

        try:
            validated_data['efficacy_rating'] = float(structured_data.get('efficacy_rating', 0))
        except ValueError:
            validated_data['efficacy_rating'] = 0

        # Add chart data
        if validated_data['toxicity_level'] > 0 or validated_data['efficacy_rating'] > 0:
            validated_data['chart_data'] = {
                'toxicity': validated_data['toxicity_level'],
                'efficacy': validated_data['efficacy_rating']
            }

        return validated_data

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error for {pesticide}: {str(e)}")
        logging.error(f"Error occurred at position: {e.pos}")
        logging.error(f"The problematic document: {e.doc}")
    except Exception as e:
        logging.error(f"Unexpected error for {pesticide}: {str(e)}")

    # Return a default structure if any error occurs
    return {
        'overview': 'Unable to retrieve information',
        'classification': 'Unknown',
        'primary_uses': [],
        'key_characteristics': [],
        'toxicity_level': 0,
        'efficacy_rating': 0,
        'chart_data': {}
    }


def get_structured_application_guide(pesticide: str) -> Dict[str, Any]:
    query = f"""Provide a detailed, farmer-friendly application guide for {pesticide} in the following JSON format:
    {{
        "overview": "Easy-to-understand overview of {pesticide} application (200-250 words)",
        "preparation": [
            "Step 1 of preparation with detailed explanation (100-150 words)",
            "Step 2 of preparation with detailed explanation (100-150 words)",
            "Step 3 of preparation with detailed explanation (100-150 words)"
        ],
        "application_method": [
            "Step 1 of application with detailed explanation (100-150 words)",
            "Step 2 of application with detailed explanation (100-150 words)",
            "Step 3 of application with detailed explanation (100-150 words)"
        ],
        "timing": "Practical information about when to apply {pesticide} (150-200 words)",
        "dosage": "Easy-to-follow dosage information with examples (150-200 words)",
        "safety_precautions": [
            "Safety precaution 1 with detailed explanation (100-150 words)",
            "Safety precaution 2 with detailed explanation (100-150 words)",
            "Safety precaution 3 with detailed explanation (100-150 words)"
        ],
        "environmental_considerations": "Farmer-friendly information about environmental impact and precautions (200-250 words)",
        "reentry_interval": "Clear explanation of safe reentry time (100-150 words)",
        "application_frequency": {{
            "frequency": "Simple explanation of how often {pesticide} can be applied (100-150 words)",
            "max_applications": "Easy-to-understand information about maximum applications per season (100-150 words)"
        }},
        "chart_data": {{
            "optimal_temperature": [min_temp, max_temp],
            "rainfall_sensitivity": rating_from_1_to_10
        }}
    }}
    Use simple, clear language. Avoid technical terms, and if used, explain them in farmer-friendly terms. 
    Provide practical, actionable information. If exact information is not available, provide the best estimate 
    or state 'Unknown' for text fields and 0 for numerical fields."""

    try:
        raw_result = get_model_response(query)
        logging.info(f"Raw result for {pesticide} application guide: {raw_result}")

        # Attempt to parse the JSON
        structured_data = json.loads(raw_result)

        # Validate and clean the structured data
        validated_data = {
            'overview': structured_data.get('overview', 'Information not available'),
            'preparation': structured_data.get('preparation', [])[:3],
            'application_method': structured_data.get('application_method', [])[:3],
            'timing': structured_data.get('timing', 'Information not available'),
            'dosage': structured_data.get('dosage', 'Information not available'),
            'safety_precautions': structured_data.get('safety_precautions', [])[:3],
            'environmental_considerations': structured_data.get('environmental_considerations', 'Information not available'),
            'reentry_interval': structured_data.get('reentry_interval', 'Information not available'),
            'application_frequency': {
                'frequency': structured_data.get('application_frequency', {}).get('frequency', 'Information not available'),
                'max_applications': structured_data.get('application_frequency', {}).get('max_applications', 'Information not available')
            },
            'chart_data': {}
        }

        # Process chart data
        chart_data = structured_data.get('chart_data', {})
        if isinstance(chart_data.get('optimal_temperature'), list) and len(chart_data['optimal_temperature']) == 2:
            validated_data['chart_data']['optimal_temperature'] = chart_data['optimal_temperature']
        else:
            validated_data['chart_data']['optimal_temperature'] = [0, 0]

        validated_data['chart_data']['rainfall_sensitivity'] = float(chart_data.get('rainfall_sensitivity', 0))

        return validated_data

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error for {pesticide} application guide: {str(e)}")
        logging.error(f"Error occurred at position: {e.pos}")
        logging.error(f"The problematic document: {e.doc}")
    except Exception as e:
        logging.error(f"Unexpected error for {pesticide} application guide: {str(e)}")

    # Return a default structure if any error occurs
    return {
        'overview': 'Unable to retrieve information',
        'preparation': [],
        'application_method': [],
        'timing': 'Information not available',
        'dosage': 'Information not available',
        'safety_precautions': [],
        'environmental_considerations': 'Information not available',
        'reentry_interval': 'Information not available',
        'application_frequency': {
            'frequency': 'Information not available',
            'max_applications': 'Information not available'
        },
        'chart_data': {
            'optimal_temperature': [0, 0],
            'rainfall_sensitivity': 0
        }
    }


def get_structured_safety_precautions(pesticide: str) -> Dict[str, Any]:
    query = f"""Provide detailed, easy-to-understand safety precautions for {pesticide} in the following JSON format:
    {{
        "overview": "Farmer-friendly overview of safety concerns for {pesticide} (200-250 words)",
        "personal_protective_equipment": [
            "PPE item 1 with detailed explanation (100-150 words)",
            "PPE item 2 with detailed explanation (100-150 words)",
            "PPE item 3 with detailed explanation (100-150 words)"
        ],
        "handling_precautions": [
            "Precaution 1 with detailed explanation (100-150 words)",
            "Precaution 2 with detailed explanation (100-150 words)",
            "Precaution 3 with detailed explanation (100-150 words)"
        ],
        "application_safety": [
            "Safety measure 1 with detailed explanation (100-150 words)",
            "Safety measure 2 with detailed explanation (100-150 words)",
            "Safety measure 3 with detailed explanation (100-150 words)"
        ],
        "storage_disposal": [
            "Storage/disposal guideline 1 with detailed explanation (100-150 words)",
            "Storage/disposal guideline 2 with detailed explanation (100-150 words)",
            "Storage/disposal guideline 3 with detailed explanation (100-150 words)"
        ],
        "environmental_safety": [
            "Environmental precaution 1 with detailed explanation (100-150 words)",
            "Environmental precaution 2 with detailed explanation (100-150 words)",
            "Environmental precaution 3 with detailed explanation (100-150 words)"
        ],
        "first_aid_measures": {{
            "skin_contact": "Clear first aid instructions for skin contact (100-150 words)",
            "eye_contact": "Clear first aid instructions for eye contact (100-150 words)",
            "inhalation": "Clear first aid instructions for inhalation (100-150 words)",
            "ingestion": "Clear first aid instructions for ingestion (100-150 words)"
        }},
        "emergency_procedures": "Easy-to-follow steps for spills or other emergencies (200-250 words)",
        "safety_ratings": {{
            "toxicity": rating_from_1_to_10,
            "flammability": rating_from_1_to_10,
            "reactivity": rating_from_1_to_10,
            "specific_hazard": "specific hazard symbol or 'N/A'"
        }},
        "reentry_interval": "Farmer-friendly explanation of safe reentry interval after application (100-150 words)"
    }}
    Use simple, clear language that farmers can easily understand. Avoid technical jargon, and if used, provide 
    simple explanations. Focus on practical, actionable safety information. If exact information is not available, 
    provide the best estimate or state 'Unknown' for text fields and 0 for numerical fields."""

    try:
        raw_result = get_model_response(query)
        logging.info(f"Raw result for {pesticide} safety precautions: {raw_result}")

        # Attempt to parse the JSON
        structured_data = json.loads(raw_result)

        # Validate and clean the structured data
        validated_data = {
            'overview': structured_data.get('overview', 'Information not available'),
            'personal_protective_equipment': structured_data.get('personal_protective_equipment', [])[:3],
            'handling_precautions': structured_data.get('handling_precautions', [])[:3],
            'application_safety': structured_data.get('application_safety', [])[:3],
            'storage_disposal': structured_data.get('storage_disposal', [])[:3],
            'environmental_safety': structured_data.get('environmental_safety', [])[:3],
            'first_aid_measures': {
                'skin_contact': structured_data.get('first_aid_measures', {}).get('skin_contact', 'Information not available'),
                'eye_contact': structured_data.get('first_aid_measures', {}).get('eye_contact', 'Information not available'),
                'inhalation': structured_data.get('first_aid_measures', {}).get('inhalation', 'Information not available'),
                'ingestion': structured_data.get('first_aid_measures', {}).get('ingestion', 'Information not available')
            },
            'emergency_procedures': structured_data.get('emergency_procedures', 'Information not available'),
            'safety_ratings': {
                'toxicity': float(structured_data.get('safety_ratings', {}).get('toxicity', 0)),
                'flammability': float(structured_data.get('safety_ratings', {}).get('flammability', 0)),
                'reactivity': float(structured_data.get('safety_ratings', {}).get('reactivity', 0)),
                'specific_hazard': structured_data.get('safety_ratings', {}).get('specific_hazard', 'N/A')
            },
            'reentry_interval': structured_data.get('reentry_interval', 'Information not available')
        }

        return validated_data

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error for {pesticide} safety precautions: {str(e)}")
        logging.error(f"Error occurred at position: {e.pos}")
        logging.error(f"The problematic document: {e.doc}")
    except Exception as e:
        logging.error(f"Unexpected error for {pesticide} safety precautions: {str(e)}")

    # Return a default structure if any error occurs
    return {
        'overview': 'Unable to retrieve information',
        'personal_protective_equipment': [],
        'handling_precautions': [],
        'application_safety': [],
        'storage_disposal': [],
        'environmental_safety': [],
        'first_aid_measures': {
            'skin_contact': 'Information not available',
            'eye_contact': 'Information not available',
            'inhalation': 'Information not available',
            'ingestion': 'Information not available'
        },
        'emergency_procedures': 'Information not available',
        'safety_ratings': {
            'toxicity': 0,
            'flammability': 0,
            'reactivity': 0,
            'specific_hazard': 'N/A'
        },
        'reentry_interval': 'Information not available'
    }


def get_structured_environmental_impact(pesticide: str) -> Dict[str, Any]:
    query = f"""Provide detailed, farmer-friendly environmental impact information for {pesticide} in the following JSON format:
    {{
        "overview": "Easy-to-understand overview of {pesticide}'s environmental impact (250-300 words)",
        "soil_impact": {{
            "description": "Farmer-friendly description of impact on soil (150-200 words)",
            "persistence": "Persistence in soil (in days)",
            "mobility": rating_from_1_to_10
        }},
        "water_impact": {{
            "description": "Clear description of impact on water bodies (150-200 words)",
            "aquatic_toxicity": rating_from_1_to_10,
            "leaching_potential": rating_from_1_to_10
        }},
        "air_impact": {{
            "description": "Farmer-friendly description of impact on air quality (150-200 words)",
            "volatility": rating_from_1_to_10
        }},
        "biodiversity_impact": {{
            "description": "Clear description of impact on biodiversity (200-250 words)",
            "bee_toxicity": rating_from_1_to_10,
            "bird_toxicity": rating_from_1_to_10,
            "mammal_toxicity": rating_from_1_to_10
        }},
        "mitigation_strategies": [
            "Strategy 1 with detailed, practical explanation (100-150 words)",
            "Strategy 2 with detailed, practical explanation (100-150 words)",
            "Strategy 3 with detailed, practical explanation (100-150 words)"
        ],
        "eco_friendly_alternatives": [
            "Alternative 1 with detailed, practical explanation (100-150 words)",
            "Alternative 2 with detailed, practical explanation (100-150 words)",
            "Alternative 3 with detailed, practical explanation (100-150 words)"
        ]
    }}
    Use simple, clear language that farmers can easily understand. Avoid technical jargon, and if used, 
    provide simple explanations. Focus on practical, actionable information about environmental impact. 
    If exact information is not available, provide the best estimate or state 'Unknown' for text fields 
    and 0 for numerical fields."""

    try:
        raw_result = get_model_response(query)
        logging.info(f"Raw result for {pesticide} environmental impact: {raw_result}")

        # Attempt to parse the JSON
        structured_data = json.loads(raw_result)

        # Validate and clean the structured data
        validated_data = {
            'overview': structured_data.get('overview', 'Information not available'),
            'soil_impact': {
                'description': structured_data.get('soil_impact', {}).get('description', 'Information not available'),
                'persistence': structured_data.get('soil_impact', {}).get('persistence', 'Unknown'),
                'mobility': float(structured_data.get('soil_impact', {}).get('mobility', 0))
            },
            'water_impact': {
                'description': structured_data.get('water_impact', {}).get('description', 'Information not available'),
                'aquatic_toxicity': float(structured_data.get('water_impact', {}).get('aquatic_toxicity', 0)),
                'leaching_potential': float(structured_data.get('water_impact', {}).get('leaching_potential', 0))
            },
            'air_impact': {
                'description': structured_data.get('air_impact', {}).get('description', 'Information not available'),
                'volatility': float(structured_data.get('air_impact', {}).get('volatility', 0))
            },
            'biodiversity_impact': {
                'description': structured_data.get('biodiversity_impact', {}).get('description', 'Information not available'),
                'bee_toxicity': float(structured_data.get('biodiversity_impact', {}).get('bee_toxicity', 0)),
                'bird_toxicity': float(structured_data.get('biodiversity_impact', {}).get('bird_toxicity', 0)),
                'mammal_toxicity': float(structured_data.get('biodiversity_impact', {}).get('mammal_toxicity', 0))
            },
            'mitigation_strategies': structured_data.get('mitigation_strategies', [])[:3],
            'eco_friendly_alternatives': structured_data.get('eco_friendly_alternatives', [])[:3]
        }

        # Add chart data
        validated_data['chart_data'] = {
            'soil_mobility': validated_data['soil_impact']['mobility'],
            'aquatic_toxicity': validated_data['water_impact']['aquatic_toxicity'],
            'leaching_potential': validated_data['water_impact']['leaching_potential'],
            'volatility': validated_data['air_impact']['volatility'],
            'bee_toxicity': validated_data['biodiversity_impact']['bee_toxicity'],
            'bird_mammal_toxicity': (validated_data['biodiversity_impact']['bird_toxicity'] + validated_data['biodiversity_impact']['mammal_toxicity']) / 2
        }

        return validated_data

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error for {pesticide} environmental impact: {str(e)}")
        logging.error(f"Error occurred at position: {e.pos}")
        logging.error(f"The problematic document: {e.doc}")
    except Exception as e:
        logging.error(f"Unexpected error for {pesticide} environmental impact: {str(e)}")

    # Return a default structure if any error occurs
    return {
        'overview': 'Unable to retrieve information',
        'soil_impact': {'description': 'Information not available', 'persistence': 'Unknown', 'mobility': 0},
        'water_impact': {'description': 'Information not available', 'aquatic_toxicity': 0, 'leaching_potential': 0},
        'air_impact': {'description': 'Information not available', 'volatility': 0},
        'biodiversity_impact': {'description': 'Information not available', 'bee_toxicity': 0, 'bird_toxicity': 0, 'mammal_toxicity': 0},
        'mitigation_strategies': [],
        'eco_friendly_alternatives': [],
        'chart_data': {
            'soil_mobility': 0,
            'aquatic_toxicity': 0,
            'leaching_potential': 0,
            'volatility': 0,
            'bee_toxicity': 0,
            'bird_mammal_toxicity': 0
        }
    }


def get_structured_residue_harvest(pesticide: str) -> Dict[str, Any]:
    query = f"""Provide detailed, farmer-friendly information about residue and harvest time for {pesticide} in the following JSON format:
    {{
        "overview": "Easy-to-understand overview of {pesticide}'s residue concerns and harvest considerations (250-300 words)",
        "half_life": {{
            "soil": {{
                "days": number_of_days,
                "explanation": "Simple explanation of half-life in soil (100-150 words)"
            }},
            "plant": {{
                "days": number_of_days,
                "explanation": "Simple explanation of half-life on/in plants (100-150 words)"
            }}
        }},
        "pre_harvest_interval": {{
            "description": "Farmer-friendly explanation of pre-harvest interval (150-200 words)",
            "intervals": [
                {{"crop": "Crop name", "days": number_of_days, "explanation": "Simple explanation (50-100 words)"}},
                {{"crop": "Crop name", "days": number_of_days, "explanation": "Simple explanation (50-100 words)"}},
                {{"crop": "Crop name", "days": number_of_days, "explanation": "Simple explanation (50-100 words)"}}
            ]
        }},
        "maximum_residue_limits": [
            {{"crop": "Crop name", "limit": number_in_ppm, "explanation": "Farmer-friendly explanation (100-150 words)"}},
            {{"crop": "Crop name", "limit": number_in_ppm, "explanation": "Farmer-friendly explanation (100-150 words)"}},
            {{"crop": "Crop name", "limit": number_in_ppm, "explanation": "Farmer-friendly explanation (100-150 words)"}}
        ],
        "factors_affecting_residue": [
            "Factor 1 with detailed, practical explanation (100-150 words)",
            "Factor 2 with detailed, practical explanation (100-150 words)",
            "Factor 3 with detailed, practical explanation (100-150 words)"
        ],
        "residue_reduction_methods": [
            "Method 1 with detailed, practical explanation (100-150 words)",
            "Method 2 with detailed, practical explanation (100-150 words)",
            "Method 3 with detailed, practical explanation (100-150 words)"
        ],
        "safety_tips": [
            "Tip 1 with detailed, practical explanation (100-150 words)",
            "Tip 2 with detailed, practical explanation (100-150 words)",
            "Tip 3 with detailed, practical explanation (100-150 words)"
        ]
    }}
    Use simple, clear language that farmers can easily understand. Avoid technical jargon, and if used, 
    provide simple explanations. Focus on practical, actionable information about residues and harvest times. 
    If exact information is not available, provide the best estimate or state 'Unknown' for text fields and 0 
    for numerical fields."""

    try:
        raw_result = get_model_response(query)
        logging.info(f"Raw result for {pesticide} residue and harvest: {raw_result}")

        # Attempt to parse the JSON
        structured_data = json.loads(raw_result)

        # Validate and clean the structured data
        validated_data = {
            'overview': structured_data.get('overview', 'Information not available'),
            'half_life': {
                'soil': {
                    'days': structured_data.get('half_life', {}).get('soil', {}).get('days', 'Unknown'),
                    'explanation': structured_data.get('half_life', {}).get('soil', {}).get('explanation', 'Information not available')
                },
                'plant': {
                    'days': structured_data.get('half_life', {}).get('plant', {}).get('days', 'Unknown'),
                    'explanation': structured_data.get('half_life', {}).get('plant', {}).get('explanation', 'Information not available')
                }
            },
            'pre_harvest_interval': {
                'description': structured_data.get('pre_harvest_interval', {}).get('description', 'Information not available'),
                'intervals': structured_data.get('pre_harvest_interval', {}).get('intervals', [])[:3]
            },
            'maximum_residue_limits': structured_data.get('maximum_residue_limits', [])[:3],
            'factors_affecting_residue': structured_data.get('factors_affecting_residue', [])[:3],
            'residue_reduction_methods': structured_data.get('residue_reduction_methods', [])[:3],
            'safety_tips': structured_data.get('safety_tips', [])[:3]
        }

        return validated_data

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error for {pesticide} residue and harvest: {str(e)}")
        logging.error(f"Error occurred at position: {e.pos}")
        logging.error(f"The problematic document: {e.doc}")
    except Exception as e:
        logging.error(f"Unexpected error for {pesticide} residue and harvest: {str(e)}")

    # Return a default structure if any error occurs
    return {
        'overview': 'Unable to retrieve information',
        'half_life': {
            'soil': {'days': 'Unknown', 'explanation': 'Information not available'},
            'plant': {'days': 'Unknown', 'explanation': 'Information not available'}
        },
        'pre_harvest_interval': {
            'description': 'Information not available',
            'intervals': []
        },
        'maximum_residue_limits': [],
        'factors_affecting_residue': [],
        'residue_reduction_methods': [],
        'safety_tips': []
    }


def general_info(pesticide):
    result = get_structured_general_info(pesticide)
    return render_template('general_info.html', pesticide=pesticide, result=result)


def application_guide():
    pesticide = request.form.get('pesticide')
    result = get_structured_application_guide(pesticide)
    return render_template('application_guide.html', pesticide=pesticide, result=result)


def safety_precautions():
    pesticide = request.form.get('pesticide')
    result = get_structured_safety_precautions(pesticide)
    return render_template('safety_precautions.html', pesticide=pesticide, result=result)


def environmental_impact():
    pesticide = request.form.get('pesticide')
    result = get_structured_environmental_impact(pesticide)
    return render_template('environmental_impact.html', pesticide=pesticide, result=result)


def residue_harvest():
    pesticide = request.form.get('pesticide')
    result = get_structured_residue_harvest(pesticide)
    return render_template('residue_harvest.html', pesticide=pesticide, result=result)


# Flask route handlers
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pesticide = request.form.get('pesticide')
        category = request.form.get('category')

        # Route to appropriate function based on the selected category
        if category == 'info':
            return general_info(pesticide)
        elif category == 'application':
            return application_guide()
        elif category == 'safety':
            return safety_precautions()
        elif category == 'environment':
            return environmental_impact()
        elif category == 'residue':
            return residue_harvest()

    return render_template('index.html')


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
