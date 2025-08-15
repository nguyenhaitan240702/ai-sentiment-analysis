// MongoDB initialization script for AI Sentiment Analysis
// Creates collections and indexes

db = db.getSiblingDB('aisent');

// Create collections with validation
db.createCollection('texts', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['source', 'lang', 'content', 'content_hash', 'created_at'],
            properties: {
                source: {
                    bsonType: 'string',
                    enum: ['api', 'batch', 'worker'],
                    description: 'Source of the text'
                },
                lang: {
                    bsonType: 'string',
                    pattern: '^[a-z]{2}$',
                    description: 'Language code'
                },
                content: {
                    bsonType: 'string',
                    minLength: 1,
                    maxLength: 10000,
                    description: 'Text content'
                },
                content_hash: {
                    bsonType: 'string',
                    description: 'SHA256 hash of content'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Creation timestamp'
                }
            }
        }
    }
});

db.createCollection('inferences', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['text_id', 'model_name', 'version', 'sentiment', 'latency_ms', 'created_at'],
            properties: {
                text_id: {
                    bsonType: 'objectId',
                    description: 'Reference to text document'
                },
                model_name: {
                    bsonType: 'string',
                    description: 'Model used for inference'
                },
                version: {
                    bsonType: 'string',
                    description: 'Model version'
                },
                sentiment: {
                    bsonType: 'object',
                    required: ['label', 'score'],
                    properties: {
                        label: {
                            bsonType: 'string',
                            enum: ['positive', 'negative', 'neutral']
                        },
                        score: {
                            bsonType: 'double',
                            minimum: 0.0,
                            maximum: 1.0
                        },
                        scores: {
                            bsonType: 'object'
                        }
                    }
                },
                latency_ms: {
                    bsonType: 'int',
                    minimum: 0,
                    description: 'Processing latency in milliseconds'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Creation timestamp'
                }
            }
        }
    }
});

db.createCollection('events', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['type', 'payload', 'created_at'],
            properties: {
                type: {
                    bsonType: 'string',
                    enum: ['api_request', 'batch_job', 'model_load', 'error'],
                    description: 'Event type'
                },
                payload: {
                    bsonType: 'object',
                    description: 'Event data'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Event timestamp'
                }
            }
        }
    }
});

// Create indexes for performance
// Texts collection indexes
db.texts.createIndex({ 'content_hash': 1 }, { unique: true });
db.texts.createIndex({ 'created_at': 1 });
db.texts.createIndex({ 'lang': 1, 'created_at': 1 });
db.texts.createIndex({ 'source': 1, 'created_at': 1 });

// Inferences collection indexes
db.inferences.createIndex({ 'text_id': 1 });
db.inferences.createIndex({ 'model_name': 1, 'created_at': 1 });
db.inferences.createIndex({ 'sentiment.label': 1, 'created_at': 1 });
db.inferences.createIndex({ 'created_at': 1 });

// Events collection indexes
db.events.createIndex({ 'type': 1, 'created_at': 1 });
db.events.createIndex({ 'created_at': 1 });

// Create TTL index for events (keep for 30 days)
db.events.createIndex({ 'created_at': 1 }, { expireAfterSeconds: 2592000 });

print('MongoDB collections and indexes created successfully');

// Insert sample data for testing
db.texts.insertOne({
    source: 'api',
    lang: 'vi', 
    content: 'Tôi rất thích sản phẩm này!',
    content_hash: 'sample_hash_1',
    created_at: new Date()
});

print('Sample data inserted successfully');
