// Neo4j Explainability Queries (deterministic, read-only)

// 1) Why would whisky hit faster?
// Compares fasted vs fed modifiers and affected absorption/emptying parameters.
MATCH (g:PopulationGroup)-[m:MODIFIES]->(p:PBPKParameter)-[:AFFECTS]->(bc:BodyCompartment)
WHERE g.group_name IN ['fasted', 'fed']
  AND p.parameter_name IN ['gastric_emptying_rate', 'intestinal_absorption_rate']
RETURN g.group_name AS group_name,
       p.parameter_name AS parameter_name,
       bc.name AS compartment,
       m.modifier AS modifier,
       m.source_dataset AS source_dataset,
       m.confidence_score AS confidence_score
ORDER BY p.parameter_name, group_name;

// 2) Why would fed state reduce BAC?
MATCH (pc:PhysiologyCondition)-[r:DECREASES]->(p:PBPKParameter)-[:AFFECTS]->(bc:BodyCompartment)
WHERE toLower(pc.condition) CONTAINS 'food' OR toLower(pc.condition) CONTAINS 'fed'
RETURN pc.condition AS condition,
       p.parameter_name AS parameter_name,
       bc.name AS compartment,
       r.source_file AS source_file,
       r.confidence_score AS confidence_score
ORDER BY pc.condition, p.parameter_name;

// 3) Which compounds contribute to hangover risk?
MATCH (b:Beverage)-[:CONTAINS]->(c:Compound)-[:CONTRIBUTES_TO]->(t:ToxicityRisk)
WHERE t.risk_type = 'hangover_amplification_modifier'
RETURN b.name AS beverage,
       c.name AS compound,
       t.risk_type AS risk_type,
       t.modifier AS modifier,
       t.source_compound_class AS source_compound_class,
       t.confidence_score AS confidence_score
ORDER BY beverage, compound;

// 4) Which enzymes metabolize beverage compounds?
MATCH (b:Beverage)-[:CONTAINS]->(c:Compound)-[:METABOLIZED_BY]->(e:Enzyme)
RETURN b.name AS beverage,
       c.name AS compound,
       e.name AS enzyme,
       e.family AS enzyme_family
ORDER BY beverage, compound, enzyme;
