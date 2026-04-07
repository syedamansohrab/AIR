package com.smartcs.indexer;

import com.smartcs.search.Searcher;
import io.javalin.Javalin;
import java.io.File;

public class Main {
    public static void main(String[] args) {
        String dataCorpusPath = "../data-corpus"; 
        String luceneIndexPath = "lucene-index"; 

        try {
            // --- 1. INDEXING PHASE (Runs once on startup) ---
            System.out.println("--- Starting Indexing Phase ---");
            DocumentIndexer indexer = new DocumentIndexer(luceneIndexPath);
            File corpusDir = new File(dataCorpusPath);
            File[] files = corpusDir.listFiles();

            if (files != null && files.length > 0) {
                for (File file : files) {
                    if (file.getName().toLowerCase().endsWith(".pdf")) {
                        indexer.indexPdfDocument(file);
                    }
                }
                System.out.println("Indexing complete!");
            }
            indexer.close();

            // --- 2. START THE API SERVER ---
            Searcher searcher = new Searcher(luceneIndexPath);
            
            Javalin app = Javalin.create(config -> {
                // Allows your Python frontend to talk to this Java backend
                config.bundledPlugins.enableCors(cors -> cors.addRule(it -> it.anyHost())); 
            }).start(8080); // Starts the server on port 8080

            System.out.println("\n🚀 SmartCS Search API is LIVE on http://localhost:8080");

            // --- 3. DEFINE THE SEARCH ENDPOINT ---
            app.get("/search", ctx -> {
                // Grab the query from the URL (e.g., /search?q=BM25)
                String query = ctx.queryParam("q"); 

                if (query == null || query.trim().isEmpty()) {
                    ctx.status(400).result("Please provide a search term using ?q=your_term");
                    return;
                }

                System.out.println("Received API request for: " + query);
                
                // Perform the search and return the results as JSON!
                ctx.json(searcher.search(query, 5)); 
            });

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}