
from pathlib import Path
from urllib.parse import urlparse
import subprocess

class CloneError(Exception):
    """Raised when repository cloning fails."""
    pass


def clone_repository(url: str, base_dir: str= 'data')-> Path:
    """
        Clone a GitHub repository into the local data directory.
        Returns:
            Path: Local path to the cloned repository.
        Raises:
            CloneError: If cloning fails.
        """
    base_path = Path(base_dir)
    base_path.mkdir(parents= True, exist_ok=True)

    parsed = urlparse(url)

    repo_name = Path(parsed.path).name
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]

    repo_path = base_path / repo_name

    git_dir = repo_path /'.git'
    if git_dir.exists():
        return repo_path

    try:
        subprocess.run(
            ['git', 'clone', url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise CloneError(f'Failed to clone repository: {e.stderr.strip()}') from e

    return repo_path