
package com.salesforce.hw

import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataReaders
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelSelector
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelsToTry._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

case class Automl_data
 (
    id: Int,
  respon_var: Int,
  Automl_feature_03: Option[String],
  Automl_feature_04: Option[Int],
  Automl_feature_05: Option[String],
  Automl_feature_06: Option[Int],
  Automl_feature_07: Option[String],
  Automl_feature_08: Option[String],
  Automl_feature_09: Option[String],
  Automl_feature_10: Option[String],
  Automl_feature_11: Option[String],
  Automl_feature_12: Option[Double],
  Automl_feature_13: Option[Double],
  Automl_feature_14: Option[Double],
  Automl_feature_15: Option[String],
  Automl_feature_16: Option[Int]
 )

object Newrr {
  def main(args: Array[String]): Unit = {
    if (args.isEmpty) {
      println("You need to pass in the CSV file path as an argument")
      sys.exit(1)
    }
    val csvFilePath = args(0)
    val csvFilePaths = args(1)
    println(s"Using user-supplied CSV file path: $csvFilePath")
    // Set up a SparkSession as normal
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName.stripSuffix("$"))
    implicit val spark = SparkSession.builder.config(conf).getOrCreate()

    val respon_var = FeatureBuilder.RealNN[Automl_data].extract(
           _.respon_var.toRealNN).asResponse
    val Automl_feature_03 = FeatureBuilder.PickList[Automl_data].extract(
           _.Automl_feature_03.map(_.toString).toPickList).asPredictor
    val Automl_feature_04 = FeatureBuilder.Integral[Automl_data].extract(
           _.Automl_feature_04.toIntegral).asPredictor
    val Automl_feature_05 = FeatureBuilder.PickList[Automl_data].extract(
           _.Automl_feature_05.map(_.toString).toPickList).asPredictor
    val Automl_feature_06 = FeatureBuilder.Integral[Automl_data].extract(
           _.Automl_feature_06.toIntegral).asPredictor
    val Automl_feature_07 = FeatureBuilder.PickList[Automl_data].extract(
           _.Automl_feature_07.map(_.toString).toPickList).asPredictor
    val Automl_feature_08 = FeatureBuilder.PickList[Automl_data].extract(
           _.Automl_feature_08.map(_.toString).toPickList).asPredictor
    val Automl_feature_09 = FeatureBuilder.PickList[Automl_data].extract(
           _.Automl_feature_09.map(_.toString).toPickList).asPredictor
    val Automl_feature_10 = FeatureBuilder.PickList[Automl_data].extract(
           _.Automl_feature_10.map(_.toString).toPickList).asPredictor
    val Automl_feature_11 = FeatureBuilder.PickList[Automl_data].extract(
           _.Automl_feature_11.map(_.toString).toPickList).asPredictor
    val Automl_feature_12 = FeatureBuilder.Real[Automl_data].extract(
           _.Automl_feature_12.toReal).asPredictor
    val Automl_feature_13 = FeatureBuilder.Real[Automl_data].extract(
           _.Automl_feature_13.toReal).asPredictor
    val Automl_feature_14 = FeatureBuilder.Real[Automl_data].extract(
           _.Automl_feature_14.toReal).asPredictor
    val Automl_feature_15 = FeatureBuilder.PickList[Automl_data].extract(
           _.Automl_feature_15.map(_.toString).toPickList).asPredictor
    val Automl_feature_16 = FeatureBuilder.Integral[Automl_data].extract(
           _.Automl_feature_16.toIntegral).asPredictor

    val Automl_dataFeatures = Seq(
          Automl_feature_03, Automl_feature_04, Automl_feature_05,
      Automl_feature_06, Automl_feature_07, Automl_feature_08,
      Automl_feature_09, Automl_feature_10, Automl_feature_11,
      Automl_feature_12, Automl_feature_13, Automl_feature_14,
      Automl_feature_15, Automl_feature_16
        ).transmogrify()


    val sanityCheck = true
    val finalFeatures = if (sanityCheck) respon_var.sanityCheck(Automl_dataFeatures) else Automl_dataFeatures
    val prediction = BinaryClassificationModelSelector.withTrainValidationSplit(
      modelTypesToUse = Seq(OpLogisticRegression)
    ).setInput(respon_var, Automl_dataFeatures).getOutput()
    val evaluator = Evaluators.BinaryClassification().setLabelCol(respon_var).setPredictionCol(prediction)
    import spark.implicits._
    val trainDataReader = DataReaders.Simple.csvCase[Automl_data](
      path = Option(csvFilePath),
      key = _.id.toString
    )
    val workflow =
      new OpWorkflow()
        .setResultFeatures(respon_var, prediction)
        .setReader(trainDataReader)
    val fittedWorkflow = workflow.train()
    val (dataframe, metrics) = fittedWorkflow.scoreAndEvaluate(evaluator = evaluator)
    println("Transformed dataframe columns:")
    dataframe.columns.foreach(println)
    println("Metrics:")
    fittedWorkflow .save("/tmp/my-model1")
    println("model_saved")
    // Load the model
    val loadedModel = workflow.loadModel("/tmp/my-model1")
    println("model_loaded")
    // Score the loaded model
    val Tpo_datassssDatas = DataReaders.Simple.csvCase[Automl_data](
      Option(csvFilePaths),
      key = _.id.toString)
    val scores = loadedModel.setReader(Tpo_datassssDatas).score()
    print("model_scored")
    scores.write.json("/tmp/my-model13")
    scores.show(true)
    println(loadedModel.summaryPretty())
  }
}
