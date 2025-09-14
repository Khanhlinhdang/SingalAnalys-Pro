import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Define components and their positions
components = {
    # Hardware Layer (y=1)
    'USRP N210/X310': {'x': 2, 'y': 1, 'color': '#1FB8CD', 'layer': 'Hardware'},
    'Ethernet/SFP': {'x': 4, 'y': 1, 'color': '#1FB8CD', 'layer': 'Hardware'},
    
    # Driver Layer (y=2)
    'UHD Driver': {'x': 3, 'y': 2, 'color': '#DB4545', 'layer': 'Driver'},
    
    # Python Interface Layer (y=3)
    'UHD Py Bind': {'x': 3, 'y': 3, 'color': '#2E8B57', 'layer': 'Python Intf'},
    
    # Application Layer (y=4)
    'USRP Intf': {'x': 1, 'y': 4, 'color': '#5D878F', 'layer': 'Application'},
    'Signal Proc': {'x': 2.5, 'y': 4, 'color': '#5D878F', 'layer': 'Application'},
    'GUI Comps': {'x': 4, 'y': 4, 'color': '#5D878F', 'layer': 'Application'},
    'Config Mgr': {'x': 5.5, 'y': 4, 'color': '#5D878F', 'layer': 'Application'},
    
    # User Interface Layer (y=5)
    'Ctrl Panels': {'x': 0.5, 'y': 5, 'color': '#D2BA4C', 'layer': 'User Interface'},
    'Spectrum': {'x': 1.75, 'y': 5, 'color': '#D2BA4C', 'layer': 'User Interface'},
    'Waterfall': {'x': 3, 'y': 5, 'color': '#D2BA4C', 'layer': 'User Interface'},
    'Constellation': {'x': 4.25, 'y': 5, 'color': '#D2BA4C', 'layer': 'User Interface'},
    'Recording': {'x': 5.5, 'y': 5, 'color': '#D2BA4C', 'layer': 'User Interface'}
}

# Define connections (from -> to)
connections = [
    ('USRP N210/X310', 'Ethernet/SFP'),
    ('Ethernet/SFP', 'UHD Driver'),
    ('UHD Driver', 'UHD Py Bind'),
    ('UHD Py Bind', 'USRP Intf'),
    ('UHD Py Bind', 'Signal Proc'),
    ('USRP Intf', 'Signal Proc'),
    ('Signal Proc', 'GUI Comps'),
    ('GUI Comps', 'Spectrum'),
    ('GUI Comps', 'Waterfall'),
    ('GUI Comps', 'Constellation'),
    ('Config Mgr', 'USRP Intf'),
    ('Config Mgr', 'Signal Proc'),
    ('Config Mgr', 'Ctrl Panels'),
    ('Ctrl Panels', 'USRP Intf'),
    ('Signal Proc', 'Recording')
]

# Create figure
fig = go.Figure()

# Add connection lines
for from_comp, to_comp in connections:
    from_pos = components[from_comp]
    to_pos = components[to_comp]
    
    fig.add_trace(go.Scatter(
        x=[from_pos['x'], to_pos['x']],
        y=[from_pos['y'], to_pos['y']],
        mode='lines',
        line=dict(color='gray', width=1.5),
        showlegend=False,
        hoverinfo='skip'
    ))

# Add components as scatter points
for name, props in components.items():
    fig.add_trace(go.Scatter(
        x=[props['x']],
        y=[props['y']],
        mode='markers+text',
        marker=dict(
            size=20,
            color=props['color'],
            symbol='square',
            line=dict(width=2, color='white')
        ),
        text=[name],
        textposition='middle center',
        textfont=dict(size=10, color='white'),
        name=props['layer'],
        legendgroup=props['layer'],
        showlegend=name == list([k for k, v in components.items() if v['layer'] == props['layer']])[0],
        hovertemplate=f"<b>{name}</b><br>Layer: {props['layer']}<extra></extra>"
    ))

# Update layout
fig.update_layout(
    title="SDR System Architecture",
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-0.5, 6.5]
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        tickmode='array',
        tickvals=[1, 2, 3, 4, 5],
        ticktext=['Hardware', 'Driver', 'Python Intf', 'Application', 'User Intf'],
        range=[0.5, 5.5],
        title='System Layers'
    ),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    ),
    plot_bgcolor='white',
    font=dict(size=12)
)

# Save the chart
fig.write_image("sdr_architecture_diagram.png", width=1200, height=800, scale=2)