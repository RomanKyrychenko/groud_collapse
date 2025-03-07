FROM python:3.11-slim

LABEL authors="romankyrychenko"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create output directory
RUN mkdir -p /app/output

# Copy the rest of the application code
COPY . .

# Create volume for output results
VOLUME /app/output

# Run the analysis and capture the output
CMD ["python", "main.py", "--input_file", "input/ground collapse.xlsx", "--input_prompt_pdf", "input/prompt.pdf"]