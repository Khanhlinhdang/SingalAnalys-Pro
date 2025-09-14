import plotly.graph_objects as go
import pandas as pd

# Create data for the pipeline blocks
block_data = {
    'x': [100, 250, 400, 550, 700, 850],
    'y': [225, 225, 225, 225, 225, 225],
    'text': ['RF Input<br>IQ', 'FFT Burst<br>Tagger', 'Tagged<br>to PDU', 'Burst<br>Downmix', 'QPSK<br>Demod', 'Frame<br>Processor'],
    'type': ['Input', 'Processing', 'Processing', 'Processing', 'Processing', 'Output'],
    'hover': ['Complex IQ Samples Input', 'Energy Detection & Tagging', 'Stream to Packet Conversion', 'Frequency Correction', 'Symbol Recovery', 'Frame Processing Output']
}

df = pd.DataFrame(block_data)

fig = go.Figure()

# Define colors for different block types
colors = {'Input': '#1FB8CD', 'Processing': '#DB4545', 'Output': '#2E8B57'}

# Add blocks as scatter points
for block_type in ['Input', 'Processing', 'Output']:
    subset = df[df['type'] == block_type]
    if not subset.empty:
        fig.add_trace(go.Scatter(
            x=subset['x'], 
            y=subset['y'],
            mode='markers+text',
            marker=dict(
                size=80, 
                color=colors[block_type], 
                symbol='square',
                line=dict(color='black', width=2)
            ),
            text=subset['text'],
            textposition='middle center',
            textfont=dict(color='white', size=10),
            name=block_type,
            hovertext=subset['hover'],
            hoverinfo='text'
        ))

# Add connection arrows using shapes
connections = [
    (150, 225, 200, 225),  # RF Input to FFT Tagger
    (300, 225, 350, 225),  # FFT Tagger to Tagged to PDU
    (450, 225, 500, 225),  # Tagged to PDU to Downmix
    (600, 225, 650, 225),  # Downmix to QPSK Demod
    (750, 225, 800, 225),  # QPSK Demod to Frame Processor
]

# Connection labels
conn_labels = ['Complex IQ', 'Tagged Stream', 'PDU', 'Baseband IQ', 'Demod Bits']

for i, (x0, y0, x1, y1) in enumerate(connections):
    # Main arrow line
    fig.add_shape(
        type="line",
        x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(color="black", width=3),
    )
    # Arrowhead - top line
    fig.add_shape(
        type="line",
        x0=x1-15, y0=y1-8, x1=x1, y1=y1,
        line=dict(color="black", width=3),
    )
    # Arrowhead - bottom line
    fig.add_shape(
        type="line",
        x0=x1-15, y0=y1+8, x1=x1, y1=y1,
        line=dict(color="black", width=3),
    )

fig.update_layout(
    title="Iridium Pipeline",
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    showlegend=True
)

fig.update_xaxes(range=[0, 950])
fig.update_yaxes(range=[150, 300])

fig.write_image("iridium_pipeline.png")