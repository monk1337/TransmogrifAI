/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op

import com.salesforce.op.utils.stages.FitStagesUtil._
import com.salesforce.op.utils.stages.FitStagesUtil
import com.salesforce.op.features.OPFeature
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.filters.FeatureDistribution
import com.salesforce.op.readers.{CustomReader, Reader, ReaderKey}
import com.salesforce.op.stages.{FeatureGeneratorStage, OPStage, OpTransformer}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.slf4j.LoggerFactory

import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Parameters for pipelines and pipeline models
 */
private[op] trait OpWorkflowCore {

  @transient protected lazy val log = LoggerFactory.getLogger(this.getClass)

  // the uid of the stage
  def uid: String

  // whether the CV/TV is performed on the workflow level
  private[op] var isWorkflowCV = false

  // the data reader for the workflow or model
  private[op] var reader: Option[Reader[_]] = None

  // final features from workflow, used to find stages of the workflow
  private[op] var resultFeatures: Array[OPFeature] = Array[OPFeature]()

  // raw features generated after data is read in and aggregated
  private[op] var rawFeatures: Array[OPFeature] = Array[OPFeature]()

  // features that have been blacklisted from use in dag
  private[op] var blacklistedFeatures: Array[OPFeature] = Array[OPFeature]()

  // map keys that were blacklisted from use in dag
  private[op] var blacklistedMapKeys: Map[String, Set[String]] = Map[String, Set[String]]()

  // raw feature distributions calculated in raw feature filter
  private[op] var rawFeatureDistributions: Array[FeatureDistribution] = Array[FeatureDistribution]()

  // stages of the workflow
  private[op] var stages: Array[OPStage] = Array[OPStage]()

  // command line parameters for the workflow stages and readers
  private[op] var parameters = new OpParams()

  private[op] def setStages(value: Array[OPStage]): this.type = {
    stages = value
    this
  }

  private[op] final def setRawFeatures(features: Array[OPFeature]): this.type = {
    rawFeatures = features
    this
  }

  private[op] final def setRawFeatureDistributions(distributions: Array[FeatureDistribution]): this.type = {
    rawFeatureDistributions = distributions
    this
  }

  /**
   * :: Experimental ::
   * Decides whether the cross-validation/train-validation-split will be done at workflow level
   * This will remove issues with data leakage, however it will impact the runtime
   *
   * @return this workflow that will train part of the DAG in the cross-validation/train validation split
   */
  @Experimental
  final def withWorkflowCV: this.type = {
    isWorkflowCV = true
    this
  }


  /**
   * Set data reader that will be used to generate data frame for stages
   *
   * @param r reader for workflow
   * @return this workflow
   */
  final def setReader(r: Reader[_]): this.type = {
    reader = Option(r)
    checkUnmatchedFeatures()
    this
  }

  /**
   * Set input dataset which contains columns corresponding to the raw features used in the workflow
   * The type of the dataset (Dataset[T]) must match the type of the FeatureBuilders[T] used to generate
   * the raw features
   *
   * @param ds  input dataset for workflow
   * @param key key extract function
   * @return this workflow
   */
  final def setInputDataset[T: WeakTypeTag](ds: Dataset[T], key: T => String = ReaderKey.randomKey _): this.type = {
    val newReader = new CustomReader[T](key) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[T], Dataset[T]] = Right(ds)
    }
    reader = Option(newReader)
    checkUnmatchedFeatures()
    this
  }

  /**
   * Set input rdd which contains columns corresponding to the raw features used in the workflow
   * The type of the rdd (RDD[T]) must match the type of the FeatureBuilders[T] used to generate the raw features
   *
   * @param rdd input rdd for workflow
   * @param key key extract function
   * @return this workflow
   */
  final def setInputRDD[T: WeakTypeTag](rdd: RDD[T], key: T => String = ReaderKey.randomKey _): this.type = {
    val newReader = new CustomReader[T](key) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[T], Dataset[T]] = Left(rdd)
    }
    reader = Option(newReader)
    checkUnmatchedFeatures()
    this
  }

  /**
   * Get the stages used in this workflow
   *
   * @return stages in the workflow
   */
  final def getStages(): Array[OPStage] = stages

  /**
   * Get the final features generated by the workflow
   *
   * @return result features for workflow
   */
  final def getResultFeatures(): Array[OPFeature] = resultFeatures

  /**
   * Get the list of raw features which have been blacklisted
   *
   * @return blacklisted features
   */
  final def getBlacklist(): Array[OPFeature] = blacklistedFeatures

  /**
   * Get the list of Map Keys which have been blacklisted
   *
   * @return blacklisted map keys
   */
  final def getBlacklistMapKeys(): Map[String, Set[String]] = blacklistedMapKeys

  /**
   * Get the parameter settings passed into the workflow
   *
   * @return OpWorkflowParams set for this workflow
   */
  final def getParameters(): OpParams = parameters

  /**
   * Get raw feature distribution information computed during raw feature filter
   * @return sequence of feature distribution information
   */
  final def getRawFeatureDistributions(): Array[FeatureDistribution] = rawFeatureDistributions

  /**
   * Determine if any of the raw features do not have a matching reader
   */
  protected def checkUnmatchedFeatures(): Unit = {
    if (rawFeatures.nonEmpty && reader.nonEmpty) {
      val readerInputTypes = reader.get.subReaders.map(_.fullTypeName).toSet
      val unmatchedFeatures = rawFeatures.filterNot(f =>
        readerInputTypes
          .contains(f.originStage.asInstanceOf[FeatureGeneratorStage[_, _ <: FeatureType]].tti.tpe.toString)
      )
      require(
        unmatchedFeatures.isEmpty,
        s"No matching data readers for ${unmatchedFeatures.length} input features:" +
          s" ${unmatchedFeatures.mkString(",")}. Readers had types: ${readerInputTypes.mkString(",")}"
      )
    }
  }

  /**
   * Check that readers and features are set and that params match them
   */
  protected def checkReadersAndFeatures() = {
    require(rawFeatures.nonEmpty, "Result features must be set")
    checkUnmatchedFeatures()

    val subReaderTypes = reader.get.subReaders.map(_.typeName).toSet
    val unmatchedReaders = subReaderTypes.filterNot { t => parameters.readerParams.contains(t) }

    if (unmatchedReaders.nonEmpty) {
      log.info(
        "Readers for types: {} do not have an override path in readerParams, so the default will be used",
        unmatchedReaders.mkString(","))
    }
  }

  /**
   * Used to generate dataframe from reader and raw features list
   *
   * @return Dataframe with all the features generated + persisted
   */
  protected def generateRawData()(implicit spark: SparkSession): DataFrame

  /**
   * Returns a dataframe containing all the columns generated up to the feature input
   *
   * @param feature             input feature to compute up to
   * @param persistEveryKStages persist data in transforms every k stages for performance improvement
   * @return Dataframe containing columns corresponding to all of the features generated before the feature given
   */
  def computeDataUpTo(feature: OPFeature, persistEveryKStages: Int = OpWorkflowModel.PersistEveryKStages)
    (implicit spark: SparkSession): DataFrame

  /**
   * Computes a dataframe containing all the columns generated up to the feature input and saves it to the
   * specified path in avro format
   */
  def computeDataUpTo(feature: OPFeature, path: String)
    (implicit spark: SparkSession): Unit = {
    val df = computeDataUpTo(feature)
    df.saveAvro(path)
  }

  /**
   * Efficiently applies all fitted stages grouping by level in the DAG where possible
   *
   * @param rawData             data to transform
   * @param dag                 computation graph
   * @param persistEveryKStages breaks in computation to persist
   * @param spark               spark session
   * @return transformed dataframe
   */
  protected def applyTransformationsDAG(
    rawData: DataFrame, dag: StagesDAG, persistEveryKStages: Int
  )(implicit spark: SparkSession): DataFrame = {
    // A holder for the last persisted rdd
    var lastPersisted: Option[DataFrame] = None
    if (dag.exists(_.exists(_._1.isInstanceOf[Estimator[_]]))) {
      throw new IllegalArgumentException("Cannot apply transformations to DAG that contains estimators")
    }

    // Apply stages layer by layer
    dag.foldLeft(rawData) { case (df, stagesLayer) =>
      // Apply all OP stages
      val opStages = stagesLayer.collect { case (s: OpTransformer, _) => s }
      val dfTransformed: DataFrame = FitStagesUtil.applyOpTransformations(opStages, df)

      lastPersisted.foreach(_.unpersist())
      lastPersisted = Some(dfTransformed)

      // Apply all non OP stages (ex. Spark wrapping stages etc)
      val sparkStages = stagesLayer.collect {
        case (s: Transformer, _) if !s.isInstanceOf[OpTransformer] => s.asInstanceOf[Transformer]
      }
      FitStagesUtil.applySparkTransformations(dfTransformed, sparkStages, persistEveryKStages)
    }
  }


  /**
   * Looks at model parents to match parent stage for features (since features are created from the estimator not
   * the fitted transformer)
   *
   * @param feature feature want to find origin stage for
   * @return index of the parent stage
   */
  protected def findOriginStageId(feature: OPFeature): Option[Int] =
    stages.zipWithIndex.collect { case (s, i) if s.getOutput().sameOrigin(feature) => i }.headOption

}