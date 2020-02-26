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
import java.util.Objects;
import java.util.PriorityQueue;

import org.apache.lucene.codecs.VectorValues;
import org.apache.lucene.document.VectorField;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
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
    return new VectorDistanceWeight(searcher, this, boost, field, queryVector, numProbes);
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
    private final IndexSearcher searcher;
    private final float boost;

    private final String field;
    private final float[] queryVector;
    private final int numProbes;

    VectorDistanceWeight(IndexSearcher searcher, Query query, float boost,
                         String field, float[] queryVector, int numProbes) {
      super(query);
      this.searcher = searcher;
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
      BinaryDocValues docValues = vectorValues.getVectorValues();
      BytesRef[] centroids = vectorValues.getCentroids();

      List<Integer> closestCentroids = findClosestCentroids(centroids);

      BooleanQuery.Builder queryBuilder = new BooleanQuery.Builder()
          .setMinimumNumberShouldMatch(1);
      for (int centroidIndex : closestCentroids) {
        Term term = new Term(field, centroids[centroidIndex]);
        queryBuilder.add(new TermQuery(term), BooleanClause.Occur.SHOULD);
      }

      BooleanQuery query = queryBuilder.build();
      Weight weight = query.createWeight(searcher, ScoreMode.COMPLETE_NO_SCORES, boost);
      Scorer scorer = weight.scorer(context);

      return new FilterScorer(scorer) {
        @Override
        public float score() throws IOException {
          int docId = docValues.advance(scorer.docID());
          assert docId != DocIdSetIterator.NO_MORE_DOCS;

          BytesRef encodedVector = docValues.binaryValue();
          double dist = VectorField.l2norm(queryVector, encodedVector);
          return (float) (boost / (1.0 + dist));
        }

        @Override
        public float getMaxScore(int upTo) throws IOException {
          return boost;
        }
      };
    }


    private static class ClusterWithDistance {
      double distance;
      int centroidId;

      public ClusterWithDistance(double distance, int centroidId) {
        this.distance = distance;
        this.centroidId = centroidId;
      }
    }

    private List<Integer> findClosestCentroids(BytesRef[] centroids) throws IOException {
      PriorityQueue<ClusterWithDistance> queue = new PriorityQueue<>(
          (first, second) -> -1 * Double.compare(first.distance, second.distance));

      for (int id = 0; id < centroids.length; id++) {
        BytesRef encodedCentroid = centroids[id];

        double dist = VectorField.l2norm(queryVector, encodedCentroid);
        if (queue.size() < numProbes) {
          queue.add(new ClusterWithDistance(dist, id));
        } else {
          ClusterWithDistance head = queue.peek();
          if (dist < head.distance) {
            queue.poll();
            queue.add(new ClusterWithDistance(dist, id));
          }
        }
      }

      List<Integer> closestCentroids = new ArrayList<>();
      for (ClusterWithDistance cluster : queue) {
        closestCentroids.add(cluster.centroidId);
      }
      return closestCentroids;
    }

    @Override
    public boolean isCacheable(LeafReaderContext ctx) {
      return true;
    }
  }
}
