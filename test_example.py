import codebleu

import codegleu.codegleu

source = """def get_file_content ( self , file_path : Path ) -> str :\n	path = self.get_repo_path() / file_path\n	with open(path, "r") as f:\n		return f.read()"""
hypothesis = source
reference = """def get_file_content ( self , file_path : Path ) -> str :\n	repo_path = self.get_repo_path()\n	relpath = ( \n		str ( file_path ).lstrip( "\\" ).lstrip( "/" )\n	) \n	path = repo_path / relpath\n	with open(path, "r") as f:\n		return f.read()"""

print(codebleu.calc_codebleu(references=[reference], predictions=[hypothesis], lang="python"))
print(codegleu.calc_codegleu(sources=[source], references=[reference], predictions=[hypothesis], lang="python", penalty=1.5))

hypothesis = """def get_file_content ( self , file_path : Path ) -> str :\n	path = self.get_repo_path() / file_path.lstrip( "\\" ).lstrip( "/" )\n	with open(path, "r") as f:\n		return f.read()"""

print(codebleu.calc_codebleu(references=[reference], predictions=[hypothesis], lang="python"))
print(codegleu.calc_codegleu(sources=[source], references=[reference], predictions=[hypothesis], lang="python", penalty=1.5))
