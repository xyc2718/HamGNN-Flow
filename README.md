# HamGNN Prediction Server and Work Flow

> ⚠️ **Warning:** This code is currently under development.

This is a Flask-based web server for performing Hamiltonian predictions, based on a refactored version of [Yang Zhong's HamGNN 2.0](https://github.com/QuantumLab-ZY/HamGNN).

It's designed to provide fast and efficient predictions by keeping the HamGNN model continuously loaded in memory. This eliminates the "cold start" time typically required for script-based predictions.

Clients can send graph data (or a path to the data) via an HTTP request and receive the Hamiltonian prediction in response.

***

This platform has been refactored into a microservices architecture to handle large-scale, automated workflows. The system is managed by a central **Orchestrator Server** that dispatches jobs to three distinct services: an **`openmxServerApi`** for preprocessing, the hot-started **`HamGNNServerAPI`** for core predictions, and a **`postprocessApi`** for tasks like band structure calculations. The entire workflow is managed asynchronously using a **Celery** task queue, enabling robust and efficient processing of large batches of jobs.

