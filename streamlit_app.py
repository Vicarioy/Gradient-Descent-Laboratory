import matplotlib.pyplot as plt
import streamlit as st

from compare import MODEL_TYPES, ComparisonExperiment, make_dataset
from visualize import (
    plot_classification_data,
    plot_loss,
    plot_optimization_path,
    plot_predictions,
)

st.set_page_config(
    page_title="Gradient descent laboratory",
    page_icon=":material/analytics:",
    layout="wide",
)

st.title("Gradient descent laboratory", icon=":material/analytics:")
st.caption("Compare models, execute optimization, and inspect the collected path.")

with st.sidebar:
    st.header("Experiment settings", icon=":material/tune:")
    model_type = st.selectbox("Model", MODEL_TYPES)
    learning_rate = st.number_input(
        "Learning rate", min_value=0.0001, value=0.1, step=0.01
    )
    epochs = st.slider("Epochs", min_value=10, max_value=2000, value=300, step=10)
    mode = st.segmented_control(
        "Update mode", ["batch", "mini_batch", "sgd"], default="mini_batch"
    )
    batch_size = st.number_input(
        "Batch size", min_value=1, value=32, step=1, disabled=mode != "mini_batch"
    )
    sample_count = st.slider("Samples", min_value=30, max_value=900, value=180, step=30)
    noise = st.slider("Noise", min_value=0.05, max_value=2.0, value=0.5, step=0.05)
    random_state = st.number_input("Random seed", min_value=0, value=7, step=1)
    execute = st.button(
        "Execute experiment",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
    )

if execute:
    X, y = make_dataset(model_type, sample_count, noise, random_state)
    experiment = ComparisonExperiment(
        X,
        y,
        model_type,
        learning_rate,
        epochs,
        batch_size,
        mode,
        random_state,
    )
    with st.status("Running optimization", expanded=False) as status:
        experiment.execute()
        status.update(label="Optimization complete", state="complete")
    st.session_state["experiment"] = experiment

experiment = st.session_state.get("experiment")
if experiment is None:
    with st.container(border=True):
        st.subheader("Ready to run", icon=":material/rocket_launch:")
        st.write("Choose settings in the sidebar, then execute an experiment.")
else:
    st.subheader("Run summary", icon=":material/query_stats:")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Model", experiment.model_type)
    metric_columns[1].metric("Initial loss", f"{experiment.history[0]:.4f}")
    metric_columns[2].metric("Final loss", f"{experiment.history[-1]:.4f}")
    metric_columns[3].metric("Score", f"{experiment.score():.1%}")

    loss_tab, path_tab, figure_tab, data_tab = st.tabs(
        ["Loss", "Optimization path", "Matplotlib", "Run data"]
    )
    with loss_tab:
        st.line_chart(experiment.loss_frame(), x="epoch", y="loss", width="stretch")
    with path_tab:
        path_frame = experiment.path_frame()
        theta_columns = [
            column for column in path_frame.columns if column.startswith("theta_")
        ]
        st.line_chart(path_frame, x="step", y=theta_columns, width="stretch")
        st.caption(f"Collected {len(path_frame):,} parameter snapshots.")
    with figure_tab:
        figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        plot_loss(experiment, axes[0])
        if experiment.model_type == "Regression":
            plot_predictions(experiment, axes[1])
        else:
            plot_classification_data(experiment, axes[1])
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        path_figure, path_axis = plt.subplots(figsize=(10, 4.5))
        plot_optimization_path(experiment, path_axis)
        path_figure.tight_layout()
        st.pyplot(path_figure)
        plt.close(path_figure)
    with data_tab:
        st.dataframe(experiment.loss_frame(), width="stretch", hide_index=True)
        st.download_button(
            "Download optimization path",
            experiment.path_frame().to_csv(index=False),
            file_name="optimization_path.csv",
            mime="text/csv",
            icon=":material/download:",
        )
