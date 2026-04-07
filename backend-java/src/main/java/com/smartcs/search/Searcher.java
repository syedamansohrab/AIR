package com.smartcs.search;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.MultiFieldQueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Searcher {
    private IndexSearcher indexSearcher;
    private DirectoryReader reader;

    public static class SearchResult {
        public String filename;
        public float score;

        public SearchResult(String filename, float score) {
            this.filename = filename;
            this.score = score;
        }
    }

    public Searcher(String indexDirectoryPath) throws IOException {
        Directory indexDirectory = FSDirectory.open(Paths.get(indexDirectoryPath));
        reader = DirectoryReader.open(indexDirectory);
        indexSearcher = new IndexSearcher(reader);
    }

    public List<SearchResult> search(String queryString, int topN) throws Exception {
        StandardAnalyzer analyzer = new StandardAnalyzer();
        
        // --- UNIT 2: ZONE INDEXING & BOOSTING ---
        // We define how much weight each field carries.
        Map<String, Float> boosts = new HashMap<>();
        boosts.put("title", 5.0f);   // 5x Multiplier if the search matches the title!
        boosts.put("content", 1.0f); // Standard 1x multiplier for body text matches.

        // MultiFieldQueryParser allows us to search across both zones simultaneously
        MultiFieldQueryParser parser = new MultiFieldQueryParser(
                new String[]{"title", "content"}, 
                analyzer, 
                boosts
        );
        
        Query query = parser.parse(queryString);

        TopDocs results = indexSearcher.search(query, topN);
        ScoreDoc[] hits = results.scoreDocs;

        List<SearchResult> responseList = new ArrayList<>();

        for (ScoreDoc hit : hits) {
            int docId = hit.doc;
            Document d = indexSearcher.doc(docId);
            responseList.add(new SearchResult(d.get("filename"), hit.score));
        }

        return responseList;
    }

    public void close() throws IOException {
        reader.close();
    }
}