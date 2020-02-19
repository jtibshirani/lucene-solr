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

package org.apache.lucene.search;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;

import org.apache.lucene.codecs.VectorValues;
import org.apache.lucene.document.VectorField;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.similarities.BooleanSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.util.BytesRef;

public class VectorDistanceQuery extends Query {

  private final String field;
  private final float[] queryVector;
  private final int numProbes;

  public VectorDistanceQuery(String field, float[] queryVector, int numProbes) {
    this.field = field;
    this.queryVector = queryVector;
    this.numProbes = numProbes;
    if (this.numProbes <= 1) {
      throw new IllegalArgumentException("The numProbes parameter must be > 1.");
    }
  }

  @Override
  public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
    return new VectorDistanceWeight(this, boost, scoreMode, field, queryVector, numProbes);
  }

  @Override
  public String toString(String field) {
    return String.format(Locale.ROOT, "VectorDistanceQuery{field=%s;fromQuery=%s;numCentroids=%d}",
        field, Arrays.toString(queryVector), numProbes);
  }

  @Override
  public void visit(QueryVisitor visitor) {
    if (visitor.acceptField(field)) {
      visitor.visitLeaf(this);
    }
  }

  @Override
  public boolean equals(Object other) {
    return sameClassAs(other) &&
        equalsTo(getClass().cast(other));
  }

  @Override
  public int hashCode() {
    return classHash() + Objects.hash(field, numProbes, queryVector);
  }

  private boolean equalsTo(VectorDistanceQuery other) {
    return Objects.equals(field, other.field) &&
        Arrays.equals(queryVector, other.queryVector) &&
        Objects.equals(numProbes, other.numProbes);
  }

  private static class VectorDistanceWeight extends Weight {
    private final ScoreMode scoreMode;
    private final float boost;

    private final String field;
    private final float[] queryVector;
    private final int numProbes;

    VectorDistanceWeight(Query query, float boost, ScoreMode scoreMode,
                         String field, float[] queryVector, int numProbes) {
      super(query);
      this.scoreMode = scoreMode;
      this.boost = boost;

      this.field = field;
      this.queryVector = queryVector;
      this.numProbes = numProbes;
    }

    @Override
    public Explanation explain(LeafReaderContext context, int doc) throws IOException {
      throw new UnsupportedOperationException();
    }

    @Override
    public Scorer scorer(LeafReaderContext context) throws IOException {
      VectorValues vectorValues = context.reader().getVectorValues(field);
      TermsEnum clusters = vectorValues.getClusterPostings().iterator();
      BinaryDocValues docValues = vectorValues.getVectorValues();

      List<BytesRef> closestCentroids = findClosestCentroids(clusters);

      List<Scorer> subScorers = new ArrayList<>();
      for (BytesRef encodedCentroid : closestCentroids) {
        boolean seekSuccess = clusters.seekExact(encodedCentroid);
        assert seekSuccess;

        Similarity.SimScorer simScorer = new BooleanSimilarity().scorer(1.0f, null);
        LeafSimScorer leafSimScorer = new LeafSimScorer(simScorer, context.reader(), field, false);
        TermScorer termScorer = new TermScorer(this, clusters.postings(null), leafSimScorer);
        subScorers.add(termScorer);
      }

      return new DisjunctionScorer(this, subScorers, scoreMode) {
        @Override
        protected float score(DisiWrapper topList) throws IOException {
          int docId = docValues.advance(topList.doc);
          assert docId == topList.doc;

          BytesRef encodedVector = docValues.binaryValue();
          double dist = VectorField.l2norm(queryVector, encodedVector);
          return (float) (boost / (1.0 + dist));
        }

        @Override
        public float getMaxScore(int upTo) {
          return boost;
        }
      };
    }

    private List<BytesRef> findClosestCentroids(TermsEnum centroids) throws IOException {
      PriorityQueue<Map.Entry<BytesRef, Double>> queue = new PriorityQueue<>(
          (first, second) -> -1 * Double.compare(first.getValue(), second.getValue()));

      while (true) {
        BytesRef encodedCentroid = centroids.next();
        if (encodedCentroid == null) {
          break;
        }

        double dist = VectorField.l2norm(queryVector, encodedCentroid);
        if (queue.size() < numProbes) {
          BytesRef centroidCopy = BytesRef.deepCopyOf(encodedCentroid);
          queue.add(Map.entry(centroidCopy, dist));
        } else {
          Map.Entry<BytesRef, Double> head = queue.peek();
          if (dist < head.getValue()) {
            queue.poll();
            BytesRef centroidCopy = BytesRef.deepCopyOf(encodedCentroid);
            queue.add(Map.entry(centroidCopy, dist));
          }
        }
      }

      List<BytesRef> closestCentroids = new ArrayList<>(queue.size());
      for (Map.Entry<BytesRef, Double> entry : queue) {
        closestCentroids.add(entry.getKey());
      }
      return closestCentroids;
    }

    @Override
    public boolean isCacheable(LeafReaderContext ctx) {
      return true;
    }
  }
}
