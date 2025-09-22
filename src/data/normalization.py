import ast
import hashlib
import libcst as cst


class NormalizeIdentifiers(ast.NodeTransformer):
    def __init__(self):
        self.var_counter = 0
        self.func_counter = 0
        self.var_names = {}
        self.func_names = {}

    def _get_var_name(self):
        name = f"var_{self.var_counter}"
        self.var_counter += 1
        return name

    def _get_func_name(self):
        name = f"func_{self.func_counter}"
        self.func_counter += 1
        return name

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)):
            if node.id not in self.var_names:
                self.var_names[node.id] = self._get_var_name()
            node.id = self.var_names[node.id]
        return self.generic_visit(node)

    def visit_arg(self, node):
        if node.arg not in self.var_names:
            self.var_names[node.arg] = self._get_var_name()
        node.arg = self.var_names[node.arg]
        return node

    def visit_FunctionDef(self, node):
        if node.name not in self.func_names:
            self.func_names[node.name] = self._get_func_name()
        node.name = self.func_names[node.name]
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        # Optional: you could normalize class names as well
        return self.generic_visit(node)


def normalize_code_to_ast_string(source_code):
    try:
        # Step 1: Parse code to AST
        tree = ast.parse(source_code)
    except SyntaxError:
        return None

    # Step 1: Parse code to AST
    tree = ast.parse(source_code)

    # Step 2: Normalize identifiers
    normalizer = NormalizeIdentifiers()
    normalized_tree = normalizer.visit(tree)
    ast.fix_missing_locations(normalized_tree)

    # Step 3: Dump normalized AST (without line numbers or column offsets)
    normalized_ast = ast.dump(normalized_tree, annotate_fields=True, include_attributes=False)
    return normalized_ast


def normalize_with_libcst(code):
    try:
        module = cst.parse_module(code)
        # You can traverse the CST like an AST, but it never crashes
        return module.code  # Returns the canonicalized version
    except Exception:
        return None
    


def code_to_hash(normalized_ast: str) -> str:
    return hashlib.md5(normalized_ast.encode('utf-8')).hexdigest()

def robust_normalize(code):
    normalized_ast = normalize_code_to_ast_string(code)
    if normalized_ast is not None:
        return code_to_hash(normalized_ast)
    
    # Try tolerant parser
    red = normalize_with_libcst(code)
    if red is not None:
        return code_to_hash(red)

    # Final fallback: hash raw source
    return code_to_hash(code)
