import regex as re
import Levenshtein
import string


class TrieNode:
    def __init__(self):
        self.children = {}
        self.regex = None
        self.is_end = False
        self.template_id = None


class LogTemplateCache:
    def __init__(self, delimiters=None):
        self.root = TrieNode()
        self.delimiters = delimiters if delimiters else [
            ' ', '(', ')', '[', ']', '{', '}', ',', ';']
        self.special_tokens = ['=']
        self.template_id = 0
        self.templates = {}
        self.template_to_template_id = {}
        self.template_id_to_logs = {}
        self.log_to_template_id = {}
        self.template_by_first_token = {}

    def verify_template(self, template):
        template = template.replace("<*>", "")
        template = template.replace(" ", "")
        return any(char not in string.punctuation for char in template)

    def check_template_token_len(self, template):
        parts = self.split_by_delimiters(template)
        cnt, threshold = 0, 4
        for part in parts:
            if any(char not in string.punctuation and char != ' ' for char in part):
                cnt += 1
        return cnt > threshold

    def split_by_delimiters(self, text, add_delimiter=True):
        result = []
        start = 0
        for i, char in enumerate(text):
            if char in self.delimiters:
                if start < i:
                    result.append(text[start:i])
                if add_delimiter:
                    result.append(char)
                start = i + 1
        if start < len(text):
            result.append(text[start:])
        return result

    def insert_template(self, template, log):
        if not self.verify_template(template):
            self.template_id += 1
            self.templates[self.template_id] = template
            self.template_to_template_id[template] = self.template_id
            self.template_id_to_logs[self.template_id] = [log]
            self.log_to_template_id[log] = self.template_id
            return

        node = self.root
        parts = self.split_by_delimiters(template)
        for part in parts:
            if '<*>' in part:
                if part not in node.children:
                    node.children[part] = TrieNode()
                    escaped_parts = [re.escape(val)
                                     for val in part.split('<*>')]
                    escaped_regex = '.*?'.join(escaped_parts)
                    node.children[part].regex = escaped_regex
                node = node.children[part]
            else:
                if part not in node.children:
                    node.children[part] = TrieNode()
                node = node.children[part]

        node.is_end = True
        self.template_id += 1
        node.template_id = self.template_id
        self.templates[self.template_id] = template
        self.template_to_template_id[template] = self.template_id
        self.template_id_to_logs[self.template_id] = [log]
        self.log_to_template_id[log] = self.template_id
        if self.template_by_first_token.get(parts[0], None) == None:
            self.template_by_first_token[parts[0]] = []
        self.template_by_first_token[parts[0]].append(self.template_id)

    def match_log(self, log):
        visited = set()

        def dfs(node, parts, index):
            if index == len(parts):
                if node.is_end:
                    return node.template_id
                return None

            if (node, index) in visited:
                return None
            visited.add((node, index))

            part = parts[index]
            if part in node.children:
                result = dfs(node.children[part], parts, index + 1)
                if result is not None:
                    return result

            for key in node.children.keys():
                if '<*>' in key:
                    child_node = node.children[key]
                    for i in range(index + 1, len(parts) + 1):
                        if parts[i-1] in self.special_tokens:
                            break
                        part_to_match = ''.join(parts[index:i])
                        if re.match(child_node.regex, part_to_match):
                            if i == len(parts) or parts[i] in child_node.children:
                                result = dfs(child_node, parts, i)
                                if result is not None:
                                    return result

            return None

        if self.log_to_template_id.get(log) is not None:
            return True

        parts = self.split_by_delimiters(log)
        template_id = dfs(self.root, parts, 0)
        if template_id is not None:
            self.log_to_template_id[log] = template_id
            return True
        return False

    def find_similar_templates(self, input_template, threshold=1):
        if not self.check_template_token_len(input_template):
            return []

        parts = self.split_by_delimiters(input_template)
        first_part = parts[0]
        similar_templates = []

        if first_part in self.template_by_first_token:
            for template_id in self.template_by_first_token[first_part]:
                existing_template = self.templates[template_id]
                input_tokens = parts
                existing_tokens = self.split_by_delimiters(
                    existing_template)
                distance = Levenshtein.distance(
                    input_tokens, existing_tokens)

                if distance > 0 and distance <= threshold:
                    log_example = self.template_id_to_logs[template_id][0]
                    similar_templates.append(
                        (existing_template, log_example, distance))

        similar_templates.sort(key=lambda x: x[2])

        if len(similar_templates) == 0:
            for template_id in self.templates.keys():
                if len(self.template_id_to_logs.get(template_id, [])) == 0:
                    continue
                existing_template = self.templates[template_id]
                input_tokens = parts
                existing_tokens = self.split_by_delimiters(
                    existing_template)
                distance = Levenshtein.distance(
                    input_tokens, existing_tokens)

                if distance > 0 and distance <= threshold:
                    log_example = self.template_id_to_logs[template_id][0]
                    similar_templates.append(
                        (existing_template, log_example, distance))

        return similar_templates

    def update_template(self, old_template, new_template, new_log):
        old_template_id = self.template_to_template_id[old_template]
        old_logs = self.template_id_to_logs.pop(old_template_id, [])

        old_parts = self.split_by_delimiters(old_template)
        self.remove_template_from_trie(old_parts)

        first_token = old_parts[0]
        if first_token in self.template_by_first_token:
            if old_template_id in self.template_by_first_token[first_token]:
                self.template_by_first_token[first_token].remove(
                    old_template_id)
            if not self.template_by_first_token[first_token]:
                del self.template_by_first_token[first_token]

        self.insert_template(new_template, new_log)
        new_template_id = self.log_to_template_id[new_log]
        self.template_id_to_logs[new_template_id].extend(old_logs)

        for log in old_logs:
            self.log_to_template_id[log] = new_template_id

    def remove_template_from_trie(self, parts):
        stack = []
        node = self.root
        for part in parts:
            if part in node.children:
                stack.append((node, part))
                node = node.children[part]
            else:
                return

        node.is_end = False
        node.template_id = None

        while stack:
            parent, part = stack.pop()
            if not parent.children[part].children and not parent.children[part].is_end:
                del parent.children[part]

    def get_template_for_log(self, log):
        return self.templates.get(self.log_to_template_id.get(log))

    def get_logs_for_template(self, template_id):
        return self.template_id_to_logs.get(template_id, [])