# AWS-Bedrock

1. Install AWS CLI and have it configured.
2. Make sure you have installed Poetry and use Python 3.12 in your environment.
   a. Install all packages in `requirements.txt` using `poetry add $(cat requirements.txt | tr '\n' ' ')`
   b. Update Poetry files: `poetry update`
3. Make sure you have access to the models used in `app.py`. If not, request access in your AWS account.
4. Copy any PDF you want to train LLM on into the `data` directory.
5. Run `app.py`:
   ```sh
   poetry run streamlit run app.py
6. Tested ollam model only. did not test the claude model( ai21.j2-mid-v1 )