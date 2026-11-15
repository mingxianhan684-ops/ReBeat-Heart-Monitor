
import gradio as gr

def monitor():
    return "ReBeat is running! DEMO MODE"

with gr.Blocks(title="ReBeat") as demo:
    gr.Markdown("# ❤️ ReBeat Heart Monitor")
    gr.Markdown("Two-Point Wearable PPG Sensor")
    gr.Markdown("Research demonstration only")

    button = gr.Button("Start Monitoring")
    output = gr.Textbox(label="System Status")

    button.click(fn=monitor, outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7860)
