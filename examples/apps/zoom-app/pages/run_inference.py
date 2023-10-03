import numpy as np
import streamlit as st
import pandas as pd
import altair as alt

from landingai.pipeline.image_source import Webcam
from landingai.predict import Predictor


if "api_key" in st.session_state and "endpoint_id" in st.session_state:
    model = Predictor(
        st.session_state["endpoint_id"], api_key=st.session_state["api_key"]
    )
    # video_src = NetworkedCamera(0, fps=1)
    image_placeholder = st.empty()
    bar_chart_placeholder = st.empty()

    pred_counts = {"Facing Camera": 0, "Facing Away": 0}
    with Webcam(fps=1) as video_src:
        for frame in video_src:
            frame.run_predict(model).overlay_predictions()
            if len(frame.frames) > 0:
                frame_with_pred = frame.frames[-1].other_images["overlay"]
                image_placeholder.empty()
                with image_placeholder.container():
                    st.image(np.array(frame_with_pred))

                if len(frame.predictions) > 0:
                    pred = frame.predictions[-1].label_name
                    pred_counts[pred] += 1
                    with bar_chart_placeholder.container():
                        data = pd.DataFrame(
                            pred_counts.items(), columns=["label", "count"]
                        )
                        chart = (
                            alt.Chart(data)
                            .mark_bar()
                            .encode(
                                x="label",
                                y="count",
                                color=alt.Color(
                                    "label",
                                    scale=alt.Scale(
                                        domain=["Facing Camera", "Facing Away"],
                                        range=["#762172", "#FFD700"],
                                    ),
                                ),
                            )
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)
else:
    st.warning("Please enter your API Key and Endpoint ID in the sidebar.")
