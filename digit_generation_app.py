import torch

import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import requests
from digit_vae_training import DigitVAE

# Configure page
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="✏️",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    """Load the trained VAE model"""
    model_path = "digit_vae_model.pth"
    # Initialize model
    model = DigitVAE(latent_dim=32)

    # Load the state dict
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
    except Exception as e:
        st.error(f"Error loading model: {e}")

    return model


def generate_images(model, digit, num_samples=5):
    """Generate images of the specified digit"""
    device = torch.device("cpu")

    # Create a random latent vector with some consistent features for the selected digit
    np.random.seed(digit)  # Ensure consistency for the digit

    # Generate different images of the same digit
    samples = []
    for i in range(num_samples):
        # Create a random latent vector
        z = torch.randn(1, model.latent_dim).to(device) * 0.5

        # Add a bias depending on the digit to guide the generation
        z[:, 0] = digit / 5.0 - 1.0  # Simple heuristic to bias towards the digit
        z[:, 1] = (digit % 5) / 2.5 - 1.0

        # Add some randomness to get diversity
        z += torch.randn_like(z) * 0.1 * (i + 1)

        # Generate the image
        with torch.no_grad():
            sample = model.decode(z)
            sample_np = sample.cpu().squeeze().numpy()
            samples.append(sample_np)

    return samples


def main():
    st.title("Handwritten Digit Generator")
    st.write("Select a digit to generate 5 handwritten versions of it!")

    # Digit selection
    digit = st.selectbox("Select a digit:", list(range(10)))

    if st.button("Generate Images"):
        with st.spinner("Generating images..."):
            # Load the model
            model = load_model()

            # Generate images
            generated_images = generate_images(model, digit)

            # Display the images
            st.subheader(f"Generated Images for Digit {digit}")

            # Create columns for displaying the images
            cols = st.columns(5)

            # Display each image in its own column
            for i, (col, img) in enumerate(zip(cols, generated_images)):
                with col:
                    st.image(img, caption=f"Sample {i+1}", width=100, clamp=True)

            st.success("Generation complete!")


if __name__ == "__main__":
    import os

    main()

    # Keep the app running
    st.markdown(
        """
    <style>
    footer {visibility: hidden;}
    </style>
    """,
        unsafe_allow_html=True,
    )
