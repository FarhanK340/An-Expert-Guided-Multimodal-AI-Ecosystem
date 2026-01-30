"""
JSON schema validation utilities.
"""

import json
from pathlib import Path
from jsonschema import validate, ValidationError


def load_schema():
    """
    Load the JSON schema definition.
    
    Returns:
        Dictionary containing the JSON schema
    """
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent.parent / 'schemas' / 'tumor_descriptor_schema.json',
        Path('schemas/tumor_descriptor_schema.json'),
        Path('./tumor_descriptor_schema.json')
    ]
    
    for schema_path in possible_paths:
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                return json.load(f)
    
    # Return embedded minimal schema if file not found
    print("WARNING: Schema file not found. Using embedded schema.")
    return get_embedded_schema()


def get_embedded_schema():
    """
    Return a minimal embedded schema for validation.
    
    Returns:
        Dictionary with basic schema structure
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "BrainTumorSegmentationDescriptor",
        "type": "object",
        "required": [
            "patient_info",
            "segmentation_results",
            "anatomical_mapping"
        ],
        "properties": {
            "patient_info": {
                "type": "object",
                "required": ["case_id"]
            },
            "segmentation_results": {
                "type": "object"
            },
            "anatomical_mapping": {
                "type": "object",
                "required": ["atlas_name", "affected_regions"]
            }
        }
    }


def validate_descriptor(descriptor_dict):
    """
    Validate a tumor descriptor against the schema.
    
    Args:
        descriptor_dict: Dictionary to validate
    
    Raises:
        ValidationError: If validation fails
    
    Returns:
        True if valid
    """
    schema = load_schema()
    
    try:
        validate(instance=descriptor_dict, schema=schema)
        return True
    except ValidationError as e:
        print(f"Validation failed: {e.message}")
        print(f"Failed path: {' -> '.join(str(p) for p in e.path)}")
        raise


def save_validated_descriptor(descriptor_dict, output_path):
    """
    Validate and save descriptor to JSON file.
    
    Args:
        descriptor_dict: Descriptor to save
        output_path: Path to save JSON file
    
    Returns:
        Path to saved file
    """
    # Validate first
    validate_descriptor(descriptor_dict)
    
    # Save with pretty printing
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(descriptor_dict, f, indent=2)
    
    print(f"Validated descriptor saved to: {output_path}")
    return output_path
