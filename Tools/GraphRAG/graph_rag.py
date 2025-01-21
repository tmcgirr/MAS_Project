

import logging
import os
import sys
import re
import asyncio
import json
from typing import Dict, List, Tuple
from uuid import uuid4
from datetime import datetime

from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.graphs import Neo4jGraph as LCNeo4jGraph
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from Levenshtein import distance as levenshtein_distance

#####################################
# CONFIG & CONSTANTS
#####################################

# VALID_ENTITIES = ["Node", "Person", "Organization"]
# VALID_RELATIONS = ["RELATED_TO", "MENTIONED_IN"]

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

#####################################
# LOGGING
#####################################

logger = logging.getLogger('graph_rag')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    log_dir = os.path.join(parent_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, 'graph_rag.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
logger.propagate = False
logger.info("GraphRAG module loaded")

#####################################
# CLASS DEFINITION
#####################################

# Example Queries
QUERY_EXAMPLES = """
    EXAMPLE USER QUERY: "Can you tell me what Oscillation is connected to?"
    EXAMPLE CYPHER QUERY: 
    // Example: List all Oscillation's connections
    MATCH (n:Node {name: "Oscillation"})-[r]->(m)
    WITH n.name as node, count(r) as connection_count
    MATCH (n:Node {name: "Oscillation"})-[r]->(m)
            RETURN 
                node,
                connection_count,
                collect({
                    connected_to: m.name,
                    relationship: type(r),
                    source_text: m.source_text
                }) as connections
            ORDER BY connection_count DESC;
            
            
            // Example USER QUERY: "What DISABLES MOT_HOVER_LEARN?"
            // Example CYPHER QUERY:
            MATCH (n:Node)-[r:DISABLES]->(target:Node {name: "MOT_HOVER_LEARN"})
            RETURN 
                n.name as source_node,
                type(r) as relationship,
                target.name as target_node,
                n.source_text as context,
                target.source_text as target_context;
                
            // Example USER QUERY: "What nodes use GENERATES_LIFT?"
            // Example CYPHER QUERY:
            MATCH (n:Node)-[r:GENERATES_LIFT]->(m:Node)
            RETURN 
                n.name as source_node,
                m.name as target_node,
                n.source_text as context
            ORDER BY n.name;
            
            // Example USER QUERY: "What is the shortest complete path between Oscillation and Motor?"
            // Example CYPHER QUERY:
            MATCH (x:Node {name: "Oscillation"}), (y:Node {name: "Motor"})
            OPTIONAL MATCH path = shortestPath((x)-[*..2]-(y))
            RETURN 
                CASE 
                    WHEN path IS NULL 
                    THEN 'No connection found within 2 steps. Would you like to search for longer paths? (This may take longer)'
                    ELSE 'Indirect connection found! Use detailed path query to see the relationship.'
                END as suggestion,
                x.source_text as from_context,
                y.source_text as to_context;
    """

class GraphRAGTool:
    """
    GraphRAGTool manages:
    - Connection to Neo4j
    - Integration with LLM for query generation and semantic search
    - Execution of fuzzy/semantic queries
    - Adding/editing/deleting nodes
    """
    def __init__(self):
        logger.info("Initializing GraphRAGTool")
        self._connect_to_graph()
        self._initialize_llms()
        self._initialize_cypher_qa_chain()
        self._ensure_indexes()

    def _connect_to_graph(self):
        """Connect to Neo4j using environment variables."""
        try:
            self.graph = Neo4jGraph(
                url=os.getenv("NEO4J_URI"),
                username=os.getenv("NEO4J_USERNAME"),
                password=os.getenv("NEO4J_PASSWORD"),
                database=os.getenv("NEO4J_DATABASE")
            )
            self.chain_graph = LCNeo4jGraph(
                url=os.getenv("NEO4J_URI"),
                username=os.getenv("NEO4J_USERNAME"),
                password=os.getenv("NEO4J_PASSWORD"),
                database=os.getenv("NEO4J_DATABASE")
            )
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {str(e)}", exc_info=True)
            raise

    def _initialize_llms(self):
        """Set up language models and embeddings."""
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm_chat = ChatOpenAI(model="gpt-4o", temperature=0)
        self.llm_extraction = ChatOpenAI(model="gpt-4o", temperature=0)

    def _initialize_cypher_qa_chain(self):
        """Initialize the Cypher-based QA chain for LLM query generation."""
        cypher_prompt = PromptTemplate(
            template="""
            Given the schema:
            Entities: {entities}
            Relationships: {relationships}
            Validation Schema: {validation_schema}
            Generate a Cypher query to answer this question: {query}.
            
            {QUERY_EXAMPLES}
            
            Include LIMIT 10 and ensure syntax is correct.
            NEVER RETURN THE EMBEDDING PROPERTY.
            """,
            input_variables=["entities", "relationships", "validation_schema", "query"]
        )

        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm_chat,
            graph=self.chain_graph,
            cypher_prompt=cypher_prompt,
            top_k=10,
            allow_dangerous_requests=True,
            verbose=True
        )

    def _ensure_indexes(self):
        """Ensure required indexes exist. Could be moved out of runtime if desired."""
        self._create_vector_index()
        self._create_basic_index()

    def _create_vector_index(self):
        query = """
        CREATE VECTOR INDEX nodeEmbeddings IF NOT EXISTS 
        FOR (n:Node) ON (n.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        try:
            self.graph.query(query)
            logger.info("Vector index verified.")
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            raise

    def _create_basic_index(self):
        query = """
        CREATE INDEX node_name_idx IF NOT EXISTS 
        FOR (n:Node) ON (n.name)
        """
        try:
            self.graph.query(query)
            logger.info("Basic name index verified.")
        except Exception as e:
            logger.error(f"Error creating basic index: {e}")
            raise

    #####################################
    # PUBLIC METHODS
    #####################################

    async def query_graph(self, query: str) -> Dict:
        """
        Process a user query:
          1. Extract terms.
          2. Find best matches (fuzzy/semantic).
          3. Generate candidate Cypher queries.
          4. Execute and gather results.
          5. Semantic fallback if needed.
          6. Combine, deduplicate, and finalize results.
        """
        logger.info("Starting graph query")
        query_terms = query.split()

        # Step 1: Find best matches for terms (fuzzy/semantic)
        fuzzy_matches = [await self._best_term_match(t) for t in query_terms if len(t) > 3]
        # Keep all match info including scores
        fuzzy_matches_info = [{"term": t, "match": m, "score": s} 
                             for t, (m, s) in zip([t for t in query_terms if len(t) > 3], fuzzy_matches)]
        
        # Filter for enhanced query
        fuzzy_matches_filtered = [m for m, s in fuzzy_matches if s > 0.6]
        enhanced_query = f"{query} {' '.join(fuzzy_matches_filtered)}"

        # Step 2: Find similar nodes by prefix/suffix methods
        similar_nodes = []
        similar_nodes_by_term = {}
        for term in query_terms:
            if len(term) > 3:
                term_matches = self._find_similar_nodes(term)
                similar_nodes.extend(term_matches)
                similar_nodes_by_term[term] = term_matches
        similar_nodes = list(set(similar_nodes))  # deduplicate node names

        # Step 3: Generate candidate Cypher queries
        candidate_queries = self._generate_cypher_queries(enhanced_query, similar_nodes)

        # Step 4: Execute Cypher queries
        cypher_results = self._execute_multiple_cypher_queries(candidate_queries)

        # Step 5: Semantic query fallback or supplement
        semantic_results = await self._semantic_query(query)

        # Combine and deduplicate
        all_results = cypher_results + semantic_results
        final_results = self._deduplicate_results(all_results)

        # Return comprehensive results including all intermediate data
        return {
            "query_info": {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "query_terms": query_terms
            },
            "term_matching": {
                "fuzzy_matches": fuzzy_matches_info,
                "similar_nodes": {
                    "all": similar_nodes,
                    "by_term": similar_nodes_by_term
                }
            },
            "queries": {
                "candidate_queries": candidate_queries,
                "executed_cypher": cypher_results,
                "semantic_results": semantic_results
            },
            "final_results": final_results,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_candidates": len(candidate_queries),
                "total_results": len(final_results)
            }
        }

    async def add_text(self, text: str) -> Dict:
        """
        Add entities from text to the graph.
        """
        logger.info("Adding text to graph")
        try:
            nodes, relationships = self._extract_and_validate_entities(text)
            self._embed_nodes(nodes)
            self._insert_nodes(nodes)
            self._insert_relationships(relationships)
            return {"status": "success", "nodes_added": len(nodes), "relationships_added": len(relationships)}
        except Exception as e:
            logger.error(f"Error adding text to graph: {str(e)}", exc_info=True)
            return {"error": str(e)}

    async def edit_node(self, node_id: str, properties: Dict) -> Dict:
        """
        Edit properties of an existing node.
        """
        logger.info(f"Editing node {node_id}")
        try:
            self.graph.query(
                """
                MATCH (n {id: $id})
                SET n += $properties
                RETURN n
                """,
                {"id": node_id, "properties": properties}
            )
            return {"status": "success", "node_id": node_id}
        except Exception as e:
            logger.error(f"Error editing node: {str(e)}", exc_info=True)
            return {"error": str(e)}

    async def delete_node(self, node_id: str) -> Dict:
        """
        Delete a node from the graph.
        """
        logger.info(f"Deleting node {node_id}")
        try:
            self.graph.query(
                """
                MATCH (n {id: $id})
                DETACH DELETE n
                """,
                {"id": node_id}
            )
            return {"status": "success", "node_id": node_id}
        except Exception as e:
            logger.error(f"Error deleting node: {str(e)}", exc_info=True)
            return {"error": str(e)}

    #####################################
    # PRIVATE HELPER METHODS: SEARCH
    #####################################

    async def _best_term_match(self, term: str) -> Tuple[str, float]:
        """Return best match for a term using semantic then fuzzy fallback."""
        semantic_results = await self._semantic_query(term)
        if semantic_results:
            best_semantic = max(semantic_results, key=lambda x: x.get("score", 0))
            if best_semantic.get("score", 0) > 0.8:
                return best_semantic["name"], best_semantic["score"]

        # Fallback to fuzzy
        candidates = self._find_similar_nodes(term)
        if not candidates:
            return "", 0.0

        return self._best_fuzzy_match(term, candidates)

    def _best_fuzzy_match(self, term: str, candidates: List[str]) -> Tuple[str, float]:
        """Return best fuzzy match from candidates."""
        term_lower = term.lower()
        best_candidate = None
        best_distance = float('inf')
        for c in candidates:
            d = levenshtein_distance(term_lower, c.lower())
            if d < best_distance:
                best_distance = d
                best_candidate = c
        max_length = max(len(term), len(best_candidate)) if best_candidate else len(term)
        similarity = 1 - (best_distance / max_length) if max_length > 0 else 0
        return best_candidate, similarity

    def _find_similar_nodes(self, term: str, limit: int = 50) -> List[str]:
        """Find nodes with similar names using prefix/suffix and containment."""
        # Parameterized query to reduce risk of injection
        params = {
            "prefix": term[:3],
            "suffix": term[-3:] if len(term) > 3 else term,
            "contains": term[:3]
        }
        cypher = """
        MATCH (n)
        WHERE n.name STARTS WITH $prefix
           OR n.name ENDS WITH $suffix
           OR n.name CONTAINS $contains
        RETURN DISTINCT n.name AS name
        LIMIT $limit
        """
        params["limit"] = limit
        results = self._execute_query(cypher, params)
        return [r["name"] for r in results if "name" in r]

    async def _semantic_query(self, query: str) -> List[Dict]:
        """Perform semantic (vector) search or fallback to keyword."""
        try:
            query_embedding = self.embedding_model.embed_query(query)
            if self._is_vector_index_online():
                return self._vector_search(query_embedding)
            else:
                return self._keyword_fallback_search(query, top_n=5)
        except Exception as e:
            logger.error(f"Semantic query error: {e}", exc_info=True)
            return []

    def _is_vector_index_online(self) -> bool:
        """Check if vector index is online."""
        idx_query = """
        SHOW INDEXES
        YIELD name, state
        WHERE name = 'nodeEmbeddings' AND state = 'ONLINE'
        RETURN name, state
        """
        result = list(self.graph.query(idx_query))
        return len(result) > 0

    def _vector_search(self, embedding: List[float]) -> List[Dict]:
        """Run vector similarity search."""
        query = """
        CALL db.index.vector.queryNodes('nodeEmbeddings', 5, $embedding)
        YIELD node, score
        RETURN 
            node.id AS id,
            node.name AS name,
            node.displayName AS displayName,
            node.description AS description,
            labels(node)[0] AS label,
            score
        ORDER BY score DESC
        LIMIT 5
        """
        results = self._execute_query(query, {"embedding": embedding})
        return results

    def _keyword_fallback_search(self, query: str, top_n: int) -> List[Dict]:
        """Simple keyword-based fallback search."""
        cypher = """
        MATCH (n:Node)
        WHERE n.name CONTAINS $query 
           OR n.displayName CONTAINS $query
        RETURN 
            n.id as id,
            n.name as name,
            n.displayName as displayName,
            n.description as description,
            labels(n)[0] as label,
            1.0 as score
        LIMIT $top_n
        """
        return self._execute_query(cypher, {"query": query, "top_n": top_n})

    #####################################
    # PRIVATE HELPER METHODS: QUERY GEN & EXECUTION
    #####################################

    def _generate_cypher_queries(self, query: str, similar_nodes: List[str]) -> List[str]:
        """Generate candidate Cypher queries using the LLM."""
        similar_nodes_str = "', '".join(similar_nodes) if similar_nodes else ''
        prompt = f"""
        Generate exactly 5 Cypher queries as a JSON array of strings. Format:
        ["MATCH (n:Node) WHERE n.name IN ['{similar_nodes_str}'] ...", 
         "QUERY2",
         "QUERY3",
         "QUERY4",
         "QUERY5"]
        
         {QUERY_EXAMPLES}
        
        Query context: {query}
        """
        response = self.llm_chat.invoke(prompt).content.strip()
        match = re.search(r'\[[\s\S]*\]', response)
        if not match:
            return []
        array_str = match.group(0)
        array_str = re.sub(r'\s+', ' ', array_str)
        try:
            queries = json.loads(array_str)
            return queries if isinstance(queries, list) else []
        except:
            logger.error("Failed to parse LLM response for queries.")
            return []

    def _execute_multiple_cypher_queries(self, queries: List[str]) -> List[Dict]:
        """Execute multiple Cypher queries and combine results."""
        results = []
        seen_queries = set()
        for cq in queries:
            if cq not in seen_queries:
                seen_queries.add(cq)
                for row in self._execute_query(cq):
                    row["cypher_query"] = cq
                    processed = self._process_cypher_result(row)
                    results.append(processed)
        return results

    def _execute_query(self, cypher_query: str, params: dict = None) -> List[Dict]:
        """Execute a parameterized Cypher query and sanitize results."""
        try:
            logger.debug(f"Executing Cypher query:\n{cypher_query}")
            raw_results = list(self.graph.query(cypher_query, params))
            return self._sanitize_results(raw_results)
        except Exception as e:
            logger.error(f"Error executing Cypher query: {str(e)}", exc_info=True)
            return []

    #####################################
    # PRIVATE HELPER METHODS: DATA PROCESSING
    #####################################

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate results based on their JSON representation."""
        unique_map = {json.dumps(r, sort_keys=True): r for r in results}
        final = list(unique_map.values())
        # Remove embeddings if any remain
        for f in final:
            f.pop('embedding', None)
        return final

    def _sanitize_results(self, results: List[Dict]) -> List[Dict]:
        """Sanitize and flatten results from Neo4j."""
        sanitized = []
        for result in results:
            clean_result = {}
            for key, value in result.items():
                if isinstance(value, dict):
                    for sub_k, sub_v in value.items():
                        if sub_k != 'embedding' and not isinstance(sub_v, (bytes, bytearray)):
                            clean_result[f"{key}.{sub_k}"] = sub_v
                else:
                    if key != 'embedding' and not isinstance(value, (bytes, bytearray)):
                        clean_result[key] = value
            sanitized.append(clean_result)
        return sanitized

    def _process_cypher_result(self, res: Dict) -> Dict:
        """Convert a raw Cypher result row into a structured output format."""
        processed = {
            'node': {
                'id': res.get('n.id'),
                'name': res.get('n.name'),
                'displayName': res.get('n.displayName'),
                'description': res.get('n.description'),
                'labels': res.get('labels(n)', [])
            },
            'relationships': [],
            'related_nodes': [],
            'source': None
        }

        # Cleanup node dict
        processed['node'] = {k: v for k, v in processed['node'].items() if v is not None}

        # Handle relationships if available
        if 'type(r)' in res and res['type(r)']:
            rel_type = res['type(r)']
            direction = 'outgoing' if 'm.id' in res else 'incoming'
            processed['relationships'].append({'type': rel_type, 'direction': direction})

            # If related node present
            if 'm.id' in res:
                related_node = {
                    'id': res.get('m.id'),
                    'name': res.get('m.name'),
                    'displayName': res.get('m.displayName'),
                    'relationship_type': rel_type
                }
                related_node = {k: v for k, v in related_node.items() if v is not None}
                processed['related_nodes'].append(related_node)

        # If source info present
        if res.get('sourceId') or res.get('sourceText'):
            processed['source'] = {
                'id': res.get('sourceId'),
                'text': res.get('sourceText')
            }

        return processed

    #####################################
    # PRIVATE HELPER METHODS: ADDING DATA
    #####################################

    def _extract_and_validate_entities(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relationships from text via LLM, then validate them.
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Extract entities and relationships as JSON."),
            HumanMessage(content=text)
        ])
        response = self.llm_extraction.invoke(prompt.format_messages())
        data = json.loads(response.content)

        nodes = data.get("Nodes", [])
        relationships = data.get("Relationships", [])
        return self._validate_entities(nodes, relationships)

    def _validate_entities(self, nodes: List[Dict], relationships: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        valid_nodes = [n for n in nodes if n.get("label") in VALID_ENTITIES]
        valid_rels = [r for r in relationships if r.get("type") in VALID_RELATIONS]
        return valid_nodes, valid_rels

    def _embed_nodes(self, nodes: List[Dict]):
        for node in nodes:
            text_rep = " ".join(f"{k}:{v}" for k, v in node.get("properties", {}).items() if v is not None)
            node["properties"]["embedding"] = self.embedding_model.embed_query(text_rep)

    def _insert_nodes(self, nodes: List[Dict]):
        for node in nodes:
            self.graph.query(
                """
                MERGE (n:Node {id: $id})
                SET n += $properties
                """,
                {"id": node["id"], "properties": node["properties"]}
            )

    def _insert_relationships(self, relationships: List[Dict]):
        for rel in relationships:
            self.graph.query(
                f"""
                MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                MERGE (a)-[r:{rel["type"]}]->(b)
                SET r += $properties
                """,
                {
                    "from_id": rel["from_id"],
                    "to_id": rel["to_id"],
                    "properties": rel.get("properties", {})
                }
            )

###################################
# ASYNC WRAPPERS FOR AGENT USE
###################################

async def execute_graph_query(query: str) -> dict:
    tool = GraphRAGTool()
    # return await tool.execute_query(query)
    return await tool.query_graph(query)

    

async def add_to_graph(text: str) -> Dict:
    tool = GraphRAGTool()
    return await tool.add_text(text)

async def edit_node_in_graph(node_id: str, properties: Dict) -> Dict:
    tool = GraphRAGTool()
    return await tool.edit_node(node_id, properties)

async def delete_node_from_graph(node_id: str) -> Dict:
    tool = GraphRAGTool()
    return await tool.delete_node(node_id)
