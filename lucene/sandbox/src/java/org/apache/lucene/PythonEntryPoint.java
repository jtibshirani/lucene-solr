/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.VectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnGraphQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;
import py4j.GatewayServer;

public class PythonEntryPoint {
  private static final String INDEX_NAME = "vector-index";
  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector";

  private VectorValues.DistanceFunction distanceFunction;

  private Directory directory;
  private IndexWriter indexWriter;
  private IndexReader indexReader;

  public static void main(String[] args) {
    GatewayServer gatewayServer = new GatewayServer(new PythonEntryPoint());
    gatewayServer.start();
    System.out.println("Gateway Server Started");
  }

  public void prepareIndex(String function) throws IOException {
    distanceFunction = VectorValues.DistanceFunction.valueOf(function);

    Path indexPath = Files.createTempDirectory(INDEX_NAME);
    directory = NIOFSDirectory.open(indexPath);

    IndexWriterConfig iwc = new IndexWriterConfig();
    iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    iwc.setCodec(Codec.forName("Lucene90"));
    indexWriter = new IndexWriter(directory, iwc);
  }

  public void indexBatch(int startId, List<List<Number>> vectors) throws IOException {
    int id = startId;
    for (List<Number> vector : vectors) {
      Document doc = new Document();
      doc.add(new StoredField(ID_FIELD, id++));

      float[] point = convertToArray(vector);
      doc.add(new VectorField(VECTOR_FIELD, point, distanceFunction));
      indexWriter.addDocument(doc);
    }

    indexWriter.flush();
  }

  public void forceMerge() throws IOException {
    indexWriter.forceMerge(1);
    indexWriter.close();
  }

  public void openReader() throws IOException {
    indexReader = DirectoryReader.open(directory);
  }

  public List<Integer> search(List<Number> query, int k, int numCands) throws IOException {
    IndexSearcher searcher = new IndexSearcher(indexReader);

    float[] value = convertToArray(query);
    KnnGraphQuery graphQuery = new KnnGraphQuery(VECTOR_FIELD, value, numCands);

    TopDocs topDocs = searcher.search(graphQuery, k);

    List<Integer> result = new ArrayList<>(k);
    for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
      Document doc = indexReader.document(scoreDoc.doc);
      IndexableField field = doc.getField(ID_FIELD);

      assert field != null;
      result.add(field.numericValue().intValue());
    }

    return result;
  }

  private float[] convertToArray(List<Number> vector) {
    float[] point = new float[vector.size()];
    for (int i = 0; i < vector.size(); i++) {
      point[i] = vector.get(i).floatValue();
    }
    return point;
  }
}
