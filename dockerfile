# Use a specific version of the Python slim image for the builder stage
FROM python:3.9-slim as builder

# Set the working directory
WORKDIR /usr/src/app

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . .

# Build the wheel
RUN python3 setup.py sdist bdist_wheel

# --- Final Stage ---

# Use a specific version of the Python slim image for the final stage
FROM python:3.9-slim

# Set the working directory
WORKDIR /usr/src/app

# Create a non-root user and group
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy the built wheel from the builder stage
COPY --from=builder /usr/src/app/dist/*.whl .

# Install the built package
RUN pip install --no-cache-dir *.whl

# Copy the entrypoint script
COPY --from=builder /usr/src/app/spaghetti/cli_inference.py .
COPY --from=builder /usr/src/app/spaghetti_checkpoint.ckpt .

# Change ownership to the non-root user
RUN chown -R appuser:appgroup /usr/src/app

# Switch to the non-root user
USER appuser

# Set the entrypoint
ENTRYPOINT ["python3", "cli_inference.py"]