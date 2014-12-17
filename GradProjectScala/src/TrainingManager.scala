import java.io.File

import spray.json._

import scala.util.{Failure, Success, Try}

/**
 * @author kento
 */
import scala.sys.process._

trait Model
trait DataSet
case class EbookData(kind:String,path:String,shape:(Int,Int)) extends DataSet
case class SDA(kind:String ,
               randomSeed:Int,
               nIn:Int,nOut:Int,
               hiddenLayerSizes:Seq[Int],
               corruptionLevels:Seq[Double],
               preLearningRates:Seq[Double],
               batchSize:Int,
               pretrainingEpochs:Seq[Int]) extends Model

case class Layer(size:Int,corruption:Double,learningRate:Double,epochs:Int){
  def toFileName:String = """s%dc%.1fl%.1fe%d""".format(size,corruption,learningRate,epochs)
}

case class Param(dst:String,dataSet:DataSet,model:Model)
object Protocols extends DefaultJsonProtocol{
  implicit val ebookDataFormat = jsonFormat3(EbookData.apply)
  //what a boilerplates...
  implicit object DataSetFormat extends RootJsonFormat[DataSet]{
    override def read(json: JsValue): DataSet = json.convertTo[EbookData]
    override def write(obj: DataSet): JsValue = obj match {
      case ebook:EbookData => ebook.toJson
    }
  }
  implicit val sdaFormat = jsonFormat9(SDA.apply)
  implicit object ModelFormat extends RootJsonFormat[Model]{
    override def write(obj: Model): JsValue = obj match{
      case sda:SDA => sda.toJson
    }
    override def read(json: JsValue): Model = json.convertTo[SDA]
  }
  implicit val paramFormat = jsonFormat3(Param.apply)
}
object TrainingManager {
  import Protocols._
  val firstLayers = for{
    nEpoch <- Seq(45)
    learningRate<- Seq(0.2,0.01)
    size <- Seq(1000,10000)
    corruption <- Seq(0.1,0.3)
  } yield Layer(size,corruption,learningRate,nEpoch)

  val secondLayers = for{
    nEpoch <- Seq(45)
    learningRate<- Seq(0.1,0.01)
    size <- Seq(500,100)
    corruption <- Seq(0.1)
  } yield Layer(size,corruption,learningRate,nEpoch)

  val lastLayers = for{
    nEpoch <- Seq(45)
    learningRate<- Seq(0.01)
    corruption <- Seq(0.1,0d)
  } yield Layer(10,corruption,learningRate,nEpoch)

  def layersToParam(layers:Seq[Layer]):Param ={
    val dst = "../models/"+layers.map(_.toFileName).mkString+".pkl"
    Param(dst,
      EbookData("ebook","../gray",(10000,40000)),
      SDA("sda",5789,10000,10,
        layers.map(_.size),
        layers.map(_.corruption),
        layers.map(_.learningRate),
        20,
        layers.map(_.epochs))
    )
  }
  val params = for {
    first <- firstLayers
    second <- secondLayers
    last <- lastLayers
  } yield layersToParam(Seq(first,second,last))

  implicit def stringIsFile(fn:String):File = new File(fn)
  def train(param:Param):Try[Stream[String]] = Try{
    Process(Seq("python","train.py",param.toJson.toString()),"../glyph/").lineStream_!
  }
  def trainAll(params:Seq[Param]) = params.iterator.map(train)
  def main(args: Array[String]) {
//    val dst = new File("../models/test").getAbsolutePath
//    val dataSet = EbookData("ebook","../gray",(10000,1000))
//    val sda = SDA("sda",5789,10000,10,Seq(1000,100),Seq(0.2,0.2),Seq(0.2,0.1),20,Seq(15,15))
//    val param = Param(dst,dataSet,sda)
//    println(param.toJson.prettyPrint)
//    Process(Seq("python","train.py",param.toJson.prettyPrint),"../glyph/").lineStream_!.foreach(println)
    for(p <- params){
      println(p.toJson.prettyPrint)
      train(p) match {
        case Success(str) => println("leaning done!")
        case Failure(e) => e.printStackTrace()
      }
    }
  }
}
