class RelationshipGraph:
    """
    A bidirectional relationship graph that stores entities and the relationships between them.
    It supports adding connections, simplifying relationship lists, extracting triplets, and
    filtering by relevant entities.
    """

    def __init__(self, language_model):
        """
        Initializes the RelationshipGraph.
        """
        self.graph = dict()
        self.language_model = language_model

    def get_graph(self):
        """
        Returns the internal graph dictionary.
        """
        return self.graph

    def add_connections(self, entity1_vals, entity2_vals, relationships):
        """
        Adds bidirectional connections between pairs of entities with specified relationships.
        """
        for i in range(len(entity1_vals)):
            ent1 = entity1_vals[i]
            ent2 = entity2_vals[i]
            rel = relationships[i]

            if ent1 not in self.graph:
                self.graph[ent1] = dict()
            if ent2 in self.graph[ent1]:
                self.graph[ent1][ent2].append(rel)
            else:
                self.graph[ent1][ent2] = [rel]

            if ent2 not in self.graph:
                self.graph[ent2] = dict()
            if ent1 in self.graph[ent2]:
                self.graph[ent2][ent1].append(rel)
            else:
                self.graph[ent2][ent1] = [rel]

    def convert_to_simple_graph(self):
        """
        Simplifies the graph by converting lists of multiple relationships between two entities
        into a single, consolidated relationship using the language model.
        """
        for entity1 in self.graph:
            for entity2 in self.graph[entity1]:
                relationships = self.graph[entity1][entity2]
                if isinstance(relationships, list) and len(relationships) > 1:
                    consolidated_rel = self.language_model.combine_relationships(relationships)
                    self.graph[entity1][entity2] = consolidated_rel
                    self.graph[entity2][entity1] = consolidated_rel
                elif isinstance(relationships, list):
                    self.graph[entity1][entity2] = relationships[0]
                    self.graph[entity2][entity1] = self.graph[entity1][entity2]

    def extract_triplets(self):
        """
        Extracts all unique (entity1, entity2, relationship) triplets from the graph.
        """
        rels = set()
        triplets = [[] for _ in range(3)]
        for entity1 in self.graph:
            for entity2 in self.graph[entity1]:
                rel = self.graph[entity1][entity2]
                if rel not in rels:
                    rels.add(rel)
                    triplets[0].append(entity1)
                    triplets[1].append(entity2)
                    triplets[2].append(rel)

        return triplets

    def extract_relevant_relationships(self, relevant_entities):
        """
        Finds all relationships that involve at least one of the specified relevant entities.
        """
        relevant_relationships = set()
        for ent in relevant_entities:
            for neighbor in self.graph[ent]:
                val = self.graph[ent][neighbor]
                if isinstance(val, list):
                    relevant_relationships.update(val)
                else:
                    relevant_relationships.add(val)

        return list(relevant_relationships)

    def __str__(self):
        """
        Provides a string representation of the graph in a readable format.
        """
        s = []
        for entity in self.graph:
            s.append(f'{entity}')
            for neighbor in self.graph[entity]:
                s.append(f'\t{neighbor}: {self.graph[entity][neighbor]}')
            s.append('\n')

        return '\n'.join(s)
