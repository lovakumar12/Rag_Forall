import os

STRUCTURE = {
    "src": {
        "config": [
            "settings.py",
            "prompts.py",
            "__init__.py"
        ],
        "core": [
            "llm_provider.py",
            "embeddings.py",
            "vectorstore.py",
            "retriever.py",
            "reranker.py",
            "cache.py",
            "pipeline.py",
            "__init__.py"
        ],
        "api": [
            "main.py",
            "schemas.py",
            "middleware.py",
            "__init__.py"
        ],
        "integration": [
            "livekit_handler.py",
            "__init__.py"
        ],
        "__init__.py": None
    },
    "scripts": [
        "setup_db.py",
        "ingest_documents.py"
    ],
    "tests": [
        "__init__.py"
    ]
}


def create_structure(base_path: str, structure: dict):
    for name, content in structure.items():
        path = os.path.join(base_path, name)

        # Folder
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)

        # Folder with files
        elif isinstance(content, list):
            os.makedirs(path, exist_ok=True)
            for file in content:
                file_path = os.path.join(path, file)
                open(file_path, "a").close()

        # Single file
        elif content is None:
            open(path, "a").close()


def main():
    project_root = os.getcwd()  # current VS Code folder
    create_structure(project_root, STRUCTURE)
    print("✅ Project structure created successfully inside existing project!")


if __name__ == "__main__":
    main()
