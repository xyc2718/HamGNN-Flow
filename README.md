# HamGNN Prediction Server

This is a Flask-based web server for performing Hamiltonian predictions, based on a refactored version of [Yang Zhong's HamGNN 2.0](https://github.com/QuantumLab-ZY/HamGNN).

It's designed to provide fast and efficient predictions by keeping the HamGNN model continuously loaded in memory. This eliminates the "cold start" time typically required for script-based predictions.

Clients can send graph data (or a path to the data) via an HTTP request and receive the Hamiltonian prediction in response.
