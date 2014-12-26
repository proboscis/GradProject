import java.io.File

import spray.json._

import scala.util.{Failure, Success, Try}

/**
 * @author kento
 */
import scala.sys.process._

trait Model{
  def toFileName:String
}
trait DataSet{
  def toFileName:String
}
case class EbookData(kind:String,path:String,shape:(Int,Int),numData:Int) extends DataSet{
  override def toFileName: String = "ebook"
}
case class Mnist(kind:String,path:String,shape:(Int,Int),numData:Int) extends DataSet{
  override def toFileName: String = "mnist"
}
case class SDA(kind:String ,
               randomSeed:Int,
               nIn:Int,nOut:Int,
               hiddenLayerSizes:Seq[Int],
               corruptionLevels:Seq[Double],
               preLearningRates:Seq[Double],
               batchSize:Int,
               pretrainingEpochs:Seq[Int]) extends Model{
  override def toFileName: String ="sda:"+(for((((a,b),c),d) <- hiddenLayerSizes.zip(corruptionLevels).zip(preLearningRates).zip(pretrainingEpochs))
  yield Layer (a,b,c,d).toFileName).mkString
}

case class Layer(size:Int,corruption:Double,learningRate:Double,epochs:Int){
  def toFileName:String = """s%dc%.1fl%.1fe%d""".format(size,corruption,learningRate,epochs)
}

case class Param(dst:String,dataSet:DataSet,model:Model){
  def toFileName:String = dataSet.toFileName+":"+model.toFileName
}
object Protocols extends DefaultJsonProtocol{
  implicit val ebookDataFormat = jsonFormat4(EbookData.apply)
  implicit val mnistDataFormat = jsonFormat4(Mnist.apply)
  //what a boilerplates...
  implicit object DataSetFormat extends RootJsonFormat[DataSet]{
    override def read(json: JsValue): DataSet = json.convertTo[EbookData]
    override def write(obj: DataSet): JsValue = obj match {
      case ebook:EbookData => ebook.toJson
      case mnist:Mnist => mnist.toJson
    }
  }
  implicit val sdaFormat = jsonFormat9(SDA.apply)
  implicit object ModelFormat extends RootJsonFormat[Model]{
    override def write(obj: Model): JsValue = obj match{
      case sda:SDA => sda.toJson
    }
    override def read(json: JsValue): Model = json match{
      case _ => throw new RuntimeException("reading a json is not supported!")
    }
  }
  import scalaz._
  import Scalaz._
  implicit val paramFormat = jsonFormat3(Param.apply)
  val ebookTest = EbookData("ebook","path should be here",(100,100),100)
  val mnistTest = Mnist("minst","path should be here", (28,28),100)
  val sdaTest = SDA("kind should be here", 1234,28*28,10,Seq(1),Seq(0.4),Seq(0.2),20,Seq(15))
  val tests = ebookTest::mnistTest::sdaTest::Nil
  def main(args: Array[String]): Unit = {
    ebookTest.toJson.prettyPrint |> println
    mnistTest.toJson.prettyPrint |> println
    sdaTest.toJson.prettyPrint |> println
  }
}
object ParamsUtil{
  def layersToSDA(nIn:Int,layers:Seq[Layer]):SDA =
    SDA("sda",5789,nIn,10,
      layers.map(_.size),
      layers.map(_.corruption),
      layers.map(_.learningRate),
      20,
      layers.map(_.epochs))


  def layersToParam(layers:Seq[Layer]):Param ={
    val dst = "../models/ebooks/"+layers.map(_.toFileName).mkString+".pkl"
    Param(dst,
      EbookData("ebook","../gray",(100,100),10000),
      SDA("sda",5789,10000,10,
        layers.map(_.size),
        layers.map(_.corruption),
        layers.map(_.learningRate),
        20,
        layers.map(_.epochs))
    )
  }
}
object EbookTrainingParams{
  import ParamsUtil._
  val firstLayers = for{
    nEpoch <- Seq(1000,100,10)
    learningRate<- Seq(0.01)
    size <- Seq(20000,30000)
    corruption <- Seq(0)
  } yield Layer(size,corruption,learningRate,nEpoch)

  val secondLayers = for{
    nEpoch <- Seq(45)
    learningRate<- Seq(0.01)
    size <- Seq(10000)
    corruption <- Seq(0)
  } yield Layer(size,corruption,learningRate,nEpoch)

  val lastLayers = for{
    nEpoch <- Seq(45)
    learningRate<- Seq(0.01)
    corruption <- Seq(0)
  } yield Layer(10,corruption,learningRate,nEpoch)

  val params = for {
    first <- firstLayers
    second <- secondLayers
    last <- lastLayers
  } yield layersToParam(Seq(first,second,last))
}

object MnistTrainingParams{
  import ParamsUtil._
  val firstLayers = for{
    nEpoch <- Seq(100)
    learningRate<- Seq(0.01)
    size <- Seq(1000)
    corruption <- Seq(0.3)
  } yield Layer(size,corruption,learningRate,nEpoch)

  val secondLayers = for{
    nEpoch <- Seq(100)
    learningRate<- Seq(0.01)
    size <- Seq(1000)
    corruption <- Seq(0.2)
  } yield Layer(size,corruption,learningRate,nEpoch)

  val thirdLayers  =  Seq(Layer(1000,0.1,0.01,100))
  val lastLayers = for{
    nEpoch <- Seq(100)
    learningRate<- Seq(0.01)
    corruption <- Seq(0.1)
  } yield Layer(10,corruption,learningRate,nEpoch)

  
  val params = for {
    first <- firstLayers
    second <- secondLayers
    third <- thirdLayers
    last <- lastLayers
  }yield{
    Param(
      "../models/mnist/"+Seq(first,second,third,last).map(_.toFileName).mkString + ".pkl",
      Mnist("mnist","../data",(28,28),40000),
      layersToSDA(28*28,Seq(first,second,third,last)))
  }
  
}
object TrainingManager {
  import Protocols._


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
    for(p <- EbookTrainingParams.params){
      println(p.toJson.prettyPrint)
      Util.writeFile(p.toJson.prettyPrint,new File("../params/"+p.toFileName+".json"))

      train(p) match {
        case Success(str) => println("leaning done!")
        case Failure(e) => e.printStackTrace()
      }

    }
  }
}
