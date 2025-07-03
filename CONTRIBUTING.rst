=======================================
Contributing to EViz
=======================================

Thank you for your interest in contributing to EViz! Whether you're reporting bugs, suggesting new features, improving documentation, or submitting code, your contributions are highly valued.

How You Can Contribute
-----------------------

**1. Report Bugs or Request Features**

- Use the GitHub Issues page: https://github.com/cacruz/eviz-dev/issues
- Please include as much detail as possible:
  - Steps to reproduce bugs
  - Screenshots or error messages
  - Desired enhancements or new features

**2. Improve Documentation**

- See a typo or outdated information in the docs? You can fix it!
- All documentation is in the `docs/` folder (reStructuredText format).
- Follow the same steps as below for submitting code contributions.

**3. Add New Features or Fix Bugs**

- Fork the repository
- Create a new branch:
  
  .. code-block:: bash

      git checkout -b feature/your-feature-name

- Make your changes and write tests if needed
- Follow our coding style (see below)
- Run tests locally to make sure nothing breaks
- Commit changes with meaningful messages
- Push your branch and open a Pull Request (PR)

Code Style and Guidelines
--------------------------

- Use **PEP8**-compliant code
- Add docstrings and comments where appropriate
- Include unit tests for new features or bug fixes
- Keep commits focused and atomic
- Use descriptive commit messages (e.g., "Add support for custom colormap in map plots")

Running Tests
-------------

Before submitting your changes, please ensure that:

- All tests pass:
  
  .. code-block:: bash

      pytest tests # or
      pytest --cov=eviz

Development Setup
-----------------

Clone the repository and set up the development environment:

.. code-block:: bash

    git clone https://github.com/cacruz/eviz-dev.git
    cd eviz-dev
    conda env create -f environment.yaml
    conda activate viz
    pip install -e .[dev]

This installs `eviz` in editable mode along with dev dependencies.

Community Guidelines
---------------------

Be respectful and constructive in your feedback and communication.

Still Have Questions?
---------------------

Open an issue or reach out through the project's GitHub Discussions if enabled.

We’re excited to work with you!

