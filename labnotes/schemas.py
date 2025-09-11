schemas = {
    "summarisation": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Summary of the article in no more than 160 characters",
            },
            "_novelty_score": {
                "type": "string",
                "enum": ["average", "high", "very high"],
                "description": "Novelty/originality score of the article",
            },
        },
        "required": ["summary", "_novelty_score"],
        "additionalProperties": False,
    },
    "summarise_enrich": {
        "type": "object",
        "title": "Article Analysis Schema",
        "description": "Schema for analyzing web articles for mountain maps startup relevance",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Article summary in no more than 500 characters",
                "maxLength": 500
            },
            "language": {
                "type": "string",
                "description": "Primary language of the article",
                "enum": ["english", "italian", "french", "german", "spanish", "portuguese", "other"]
            },
            "location": {
                "type": "string",
                "description": "Geographical location relevant to the article",
                "enum": [
                    "Abruzzo",
                    "Basilicata",
                    "Calabria",
                    "Campania",
                    "Emilia-Romagna",
                    "Friuli-Venezia Giulia",
                    "Lazio",
                    "Liguria",
                    "Lombardia",
                    "Marche",
                    "Molise",
                    "Piemonte",
                    "Puglia",
                    "Sardegna",
                    "Sicilia",
                    "Toscana",
                    "Trentino-Alto Adige",
                    "Umbria",
                    "Valle d’Aosta",
                    "Veneto",
                    "Germania",
                    "Austria",
                    "Francia",
                    "Svizzera",
                    "Spagna",
                    "Nord America",
                    "other",
                    "not applicable"
                ],
            },
            "relevance_score": {
                "type": "string",
                "description": "How relevant the article is for a mountain maps startup",
                "enum": ["low", "medium", "high"]
            },
            "relevance_reasoning": {
                "type": "string",
                "description": "Detailed reasoning for the relevance score assignment",
                "minLength": 10,
                "maxLength": 500
            },
            "sport_type": {
                "type": "string",
                "description": "Type of sport relevant to the article",
                "enum": ["alpinism", "climbing", "hiking", "train running", "alpine skiing", "other", "not applicable"]
            },
            "topics": {
                "type": "array",
                "description": "Relevant topic tags for the article",
                "items": {
                    "type": "string",
                    "minLength": 2,
                    "maxLength": 50
            },
            "minItems": 1,
            "maxItems": 10,
            }
        },
        "required": [
            "summary",
            "language", 
            "location",
            "relevance_score",
            "relevance_reasoning",
            "sport_type",
            "topics"
        ],
        "additionalProperties": False
    }
}