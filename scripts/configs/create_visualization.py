import json
import plotly.graph_objects as go
import sys
import os

def create_semantic_sunburst_chart():
    """
    Loads semantic group data from a JSON file and generates an interactive 
    sunburst chart visualization using Plotly.
    """
    file_path = '/home/nmishra/data/storage_hpc_nishant/EL_gen/umls_datasets/scripts/configs/semantic_group_map.json'

    # Validate that the data file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found in the current directory.", file=sys.stderr)
        print("Please ensure the JSON data is saved in the same folder as this script.", file=sys.stderr)
        sys.exit(1)

    # Load the JSON data from the file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize lists for the sunburst chart components
    ids = []
    labels = []
    parents = []
    values = []

    # --- Data Preparation ---
    semantic_groups = data.get('semantic_groups', {})
    
    # Iterate through each semantic group (the main categories)
    for group_code, group_data in semantic_groups.items():
        group_name = group_data.get('name', 'Unknown Group')
        
        # In the provided file, the 'MISC' group's total is incorrect.
        # We recalculate it here by summing its children for an accurate chart.
        if group_code == 'MISC':
            group_total_percentage = sum(t.get('percentage', 0) for t in group_data.get('types', {}).values())
        else:
            group_total_percentage = group_data.get('total_percentage', 0)

        # Add data for the parent segment (inner ring)
        ids.append(group_code)
        labels.append(f"<b>{group_name} ({group_code})</b>") # Make parent label bold
        parents.append("")  # An empty parent signifies a root node
        values.append(group_total_percentage)

        # Iterate through the types within each group (the subtypes)
        types_data = group_data.get('types', {})
        for type_code, type_info in types_data.items():
            type_name = type_info.get('name', 'Unknown Type')
            type_percentage = type_info.get('percentage', 0)
            
            # Add data for the child segment (outer ring)
            ids.append(f"{group_code}-{type_code}")
            labels.append(type_name)
            parents.append(group_code)
            values.append(type_percentage)

    # --- Chart Creation ---
    fig = go.Figure()

    fig.add_trace(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",  # Children values sum up to the parent value
        hoverinfo="label+percent entry+percent parent",
        hovertemplate='<b>%{label}</b><br>Overall Percentage: %{value:.2f}%<br>Share of Parent Group: %{percentParent:.1%}',
        marker=dict(
            # Using a vivid and distinct color scale for clarity
            colorscale='Rainbow'
        ),
        textfont={'size': 14},
        insidetextorientation='radial'
    ))

    # Update the layout for a professional and readable presentation
    fig.update_layout(
        title_text="<b>Hierarchical Distribution of Medical Semantic Types</b>",
        title_x=0.5, # Center the title
        margin=dict(t=60, l=10, r=10, b=20),
        height=850,
        width=850,
        font=dict(
            family="Arial, sans-serif",
            size=12
        )
    )

    # Show the figure in a browser or save to an HTML file
    fig.show()

if __name__ == '__main__':
    create_semantic_sunburst_chart()