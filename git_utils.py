import os
import re
import shutil
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Set

import git
import numpy as np
from git.exc import GitCommandError


def is_dir_greater_than(start_path=".", size=1024):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(os.getcwd(), dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

                if total_size > size:
                    return True

    return False


class GitRepo:
    def __init__(
        self,
        repo_user: str,
        repo_name: str,
        experiments_path: str | Path = "./experiments",
    ) -> None:
        self.repo_user = repo_user
        self.repo_name = repo_name
        self.experiments_path = Path(experiments_path)
        self.repos_path = Path(experiments_path) / "repos"
        self.repo = self.clone_repo()
        try:
            self.reset()
        except Exception:
            self.repo = self.clone_repo(force_reclone=True)
            self.reset()

    def get_repo_id(self) -> str:
        return f"{self.repo_user}/{self.repo_name}"

    def get_repos_dir(self) -> Path:
        return self.repos_path

    def get_repo_path(self) -> Path:
        return self.get_repos_dir() / self.repo_user / self.repo_name

    def clone_repo(self, force_reclone=False) -> git.Repo:
        url = self.get_git_url()

        repos_dir = self.get_repos_dir()
        if not os.path.isdir(repos_dir):
            repos_dir.mkdir(parents=True, exist_ok=True)
        path = self.get_repo_path()

        if os.path.isdir(path) and is_dir_greater_than(path, 1024) and not force_reclone:  # must be some weird remnants if smaller than 1kb
            try:
                repo = git.Repo(path)
                return repo
            except Exception as e:
                print(e)
                if os.name == "nt":
                    # on windows add ? to prevent too long path length only on windows...
                    shutil.rmtree(r"\\?\ ".strip() + str(path))

        print("Cloning repo...")
        repo = git.Repo.clone_from(
            url=url,
            to_path=path,
            allow_unsafe_options=True,
            multi_options=["-c core.longpaths=true"],
        )
        print("Cloned repo to " + str(path))
        return repo

    def get_git_url(self) -> str:
        return f"https://github.com/{self.repo_user}/{self.repo_name}"

    def apply_patch(
        self,
        patch: Optional[str],
        relative_patch_file_path: Optional[str | Path] = None,
    ) -> None:
        if patch == "" or patch is None or (isinstance(patch, (int, float)) and np.isnan(patch)):
            print("Patch is empty. Skipping apply_patch...")
            return

        repo_path = self.get_repo_path()
        # relative to cloned repository root
        relative_patch_file_path = Path(relative_patch_file_path) if relative_patch_file_path is not None else self.get_default_patch_file_path()
        # absolute but relative to pepperoni repository root. We have to put the patch file into the cloned repository because git apply expects the paths relative to the cloned repo.
        absolute_patch_file_path = repo_path / relative_patch_file_path

        try:
            # make sure that the patch path exists
            absolute_patch_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(absolute_patch_file_path, "w+", newline="\n") as patch_file:
                patch_file.write(patch + "\n")

            # as said above: we have to pass the relative path here because git apply expects the paths relative to the cloned repo.
            self.repo.git.execute(
                [
                    "git",
                    "apply",
                    "--whitespace=fix",
                    str(relative_patch_file_path),
                ]
            )
        except git.exc.GitCommandError as e:
            raise PatchApplicationException(e)
        finally:
            self.try_del_file(absolute_patch_file_path)

    def get_default_patch_file_path(self) -> Path:
        return Path("tmp.diff")

    def is_nan(self, value: Any) -> bool:
        try:
            return np.isnan(value)
        except TypeError:
            return False

    def try_del_file(self, fname: Path | str) -> None:
        fpath = self.get_repo_path() / Path(fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

    """
        This returns file paths in the format repos/myuser/myrepo/path/to/my/file.py
    """

    def find_absolute_file_paths_from_patch(self, patch: str) -> Set[Path]:
        repo_path = self.get_repo_path()
        paths = self.find_file_paths_in_repo_from_patch(patch)
        return {repo_path / path for path in paths}

    """
        This returns file paths in the format path/to/my/file.py as a relative path with respect to the root folder of this git repository.
        If you want to read the content of those files you can call get_file_content(relative_path).
    """

    def find_file_paths_in_repo_from_patch(self, patch: str) -> Set[Path]:
        regex = r"\+\+\+ b\/[\S]+.[\S]+\n"

        matches = re.findall(regex, patch)
        paths = [match.replace("+++ b/", "") for match in matches]
        return {Path(path.replace("\n", "")) for path in paths}

    def get_file_content(self, file_path: Path) -> str:
        repo_path = self.get_repo_path()
        relpath = str(file_path).lstrip("\\").lstrip("/")  # if you attempt to combine two absolute pathlib paths, one is taken as a source
        path = repo_path / relpath
        with open(path, "r") as f:
            return f.read()

    def count_lines(self, file_path: Path) -> int:
        repo_path = self.get_repo_path()
        relpath = str(file_path).lstrip("\\").lstrip("/")  # if you attempt to combine two absolute pathlib paths, one is taken as a source
        path = repo_path / relpath
        with open(Path(path)) as f:
            return sum(1 for _ in f)

    def write_code(self, code: dict[str, str]) -> None:
        for file_path, content in code.items():
            self.write_file(Path(file_path), content)

    def write_file(self, file_path: Path, content: str) -> None:
        repo_path = self.get_repo_path()
        relpath = str(file_path).lstrip("\\").lstrip("/")  # if you attempt to combine two absolute pathlib paths, one is taken as a source
        path = repo_path / relpath
        if not os.path.exists(os.path.dirname(str(path))):
            os.makedirs(os.path.dirname(str(path)))
        if os.path.exists(path):
            os.chmod(path, 0o666)

        with open(path, "w+") as f:
            f.write(content)

    def get_git_diff(self) -> str:
        try:
            self.repo.git.add(all=True)
            return self.repo.git.diff(self.repo.head.commit.tree)
        finally:
            self.repo.git.execute(["git", "restore", "--staged", "."])

    def reset_to_base_commit(self, base_commit: str) -> None:
        try:
            self.reset()
            self.repo.git.fetch("--all")  # Fetch all branches and tags from remote
            self.repo.git.checkout(base_commit)
        except GitCommandError as e:
            print(f"Error checking out commit {base_commit}: {e}")
            # Handle the exception as needed (log, raise, etc.)
            # raise

    def reset(self) -> None:
        self.repo.git.execute(["git", "clean", "-df"])
        self.repo.git.checkout(".")

    def remove(self, throw_if_fails=False) -> None:
        try:
            self.repo.close()
            repo_path = self.get_repo_path()

            if os.path.isdir(repo_path):
                for root, dirs, files in os.walk(repo_path):
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        if os.path.exists(dir_path):
                            os.chmod(dir_path, stat.S_IRWXU)
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            os.chmod(file_path, stat.S_IRWXU)

                if os.name == "nt":
                    # on windows add ? to prevent too long path length...
                    shutil.rmtree("\\\\?\\" + str(repo_path.absolute()))
                else:
                    shutil.rmtree(repo_path)
        except Exception as e:
            if throw_if_fails:
                raise e
            else:
                print("Warning: Removing git repo failed...")
                print(e)


class PatchApplicationException(Exception):
    def __init__(self, original_exception: Exception):
        super().__init__("Error applying patch")
        self.original_exception = original_exception

    def __str__(self):
        return f"{super().__str__()}: {str(self.original_exception)}"


def create_git_repo(user_and_repo: str, experiments_path: str | Path = "./experiments") -> GitRepo:
    user, repo = user_and_repo.split("/")
    return GitRepo(user, repo, experiments_path)


def reset_timezone(time):
    return time.replace(tzinfo=None)


def get_timestamp_of_commit(git: git.Repo.git, git_log_parameters: List[str]):
    # from https://stackoverflow.com/questions/22497597/how-to-get-the-last-modification-date-of-a-file-in-a-git-repo
    commit_timestamp = git.log(git_log_parameters)
    commit_timestamp = int(commit_timestamp)
    commit_timestamp = datetime.fromtimestamp(commit_timestamp)
    # necessary to compare with other timestamps without a timezone
    # TODO: check if tz matters
    commit_timestamp = reset_timezone(commit_timestamp)

    return commit_timestamp
