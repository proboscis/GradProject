/**
 * @author kentomasui
 */

import java.io._
import java.net.URL

import net.coobird.thumbnailator.Thumbnails
import net.coobird.thumbnailator.filters.ImageFilter
import net.coobird.thumbnailator.resizers.configurations.ScalingMode

import scala.util.Try
import scalaz._
import Scalaz._
import scalaz.effect.IO
import sys.process._
object Util{
  def using[C<:Closeable,T](res:C)(f:C=>T):T= {
    try {
      f(res)
    } finally {
      res.close()
    }
  }
  def save(url:URL,dst:File) = using(new BufferedInputStream(url.openStream())){
    in => using(new FileOutputStream(dst)){
      out => Iterator.continually(in.read()).takeWhile(_ != -1).foreach(out.write)
    }
  }

  def writeFile(string:String,dst:File) = {
    dst.getParentFile.mkdirs()
    using(new PrintWriter(new BufferedOutputStream(new FileOutputStream(dst)))){
      out => out.write(string)
    }
  }

  def ensureDownload(validator:File=>Boolean)(dst:File,url:URL)=Try{
    validator(dst) ? () | save(url,dst)
  }
  def downloadAllAt(dir:String)(validator:File=>Boolean)(targets:Iterator[(String,URL)]) = {
    new File(dir).mkdirs()
    targets.parMap {
        case (filename, url) =>
          val dst = new File(s"$dir/$filename")
          ensureDownload(validator)(dst, url)
    }
  }
  implicit class IteratorOps[T](val itr:Iterator[T]) extends AnyVal{
    def parMap[U](f:T=>U):Iterator[U]=
      itr.grouped(100).map{_.toSeq.par.map(f).toIterator}.flatten
  }

  def throwableToStackTrace(e:Throwable):String =
    new StringWriter() <| (sw=>e.printStackTrace(new PrintWriter(sw))) |> (_.toString)

  def main(args: Array[String]) {
    val targets = Seq(
      "test"-> new URL("http://www.goole.com")
    )
    downloadAllAt("thumbnails")(_.length() > 0)(targets.iterator).foreach(println)
  }
  /**
   * iterator that returns each line of file and closes the file after iteration.
   * @param filename name of the file in any form
   * @return iterator of the lines of file.
   */
  def fileLines(filename:String):Iterator[String] = new Iterator[String]{
    val src = scala.io.Source.fromFile(filename)
    val lines = src.getLines()
    var closed = false
    override def hasNext: Boolean = (!closed) && lines.hasNext
    override def next(): String = {
      val line = lines.next()
      if(!lines.hasNext){
        src.close()
        closed = true
      }
      line
    }
  }
  def fileChars(filename:String):Iterator[Char] = new Iterator[Char]{
    val src = scala.io.Source.fromFile(filename)
    var closed = false

    override def hasNext: Boolean = (!closed) && src.hasNext
    override def next(): Char = {
      val c = src.next()
      if(!src.hasNext){
        src.close()
        closed = true
      }
      c
    }
  }
}

object EBookDownloader{
  import Util._
  def main(args: Array[String]): Unit = {
    //downloadThumbnails()
    //resizeThumbnails(28,28)(new File("thumbnails"),new File("resized/28x28"))
    val thumbnailsDir = "thumbnails"
    val comicsDir = "filtered/comics"
    val resizedDir = "resized/comics/28x28"
    copyFilteredThumbnailsTo(comicFilter)(thumbnailsDir,comicsDir)
    resizeThumbnails(28,28)(new File(comicsDir),new File(resizedDir))
  }

  import org.apache.commons.io.FileUtils._

  def copyFilteredThumbnailsTo(filter:Map[String,String]=>Boolean)(srcDir:String,dstDir:String): Unit ={
    new File(dstDir).mkdirs()
    println("copying to ",dstDir)
    books.filter(filter).foreach{
      book =>
        copyFile(
          new File(srcDir+"/"+book("SKU")+".jpg"),
          new File(dstDir+"/"+book("SKU")+".jpg")
          )
    }
    println("copy done")
  }

  def comics = books.filter(comicFilter)
  def comicFilter:String Map String => Boolean = _("大ジャンル").contains("コミック")
  def books = {
    val lines = scala.io.Source.fromFile("ebookdata/bookinfo.csv").getLines()
    val columnIndex = lines.next().split(",").zipWithIndex.toMap
    lines.map{line => columnIndex.mapValues(line.split(",").orElse{case _=>"null"})}
  }
  def downloadThumbnails(): Unit ={
    val thumbnails = books.map{
      info => Try {
        val url = info("サムネイル") + info("表示画像")
        val ext = url.split('.').last
        (info("SKU") + "." + ext) -> new URL(url)
      }
    }.filter{
      case util.Success(pair)=> true
      case util.Failure(f)=> f.printStackTrace();false
    }.map{
      _.get
    }
    Util.downloadAllAt("thumbnails")(_.length > 0)(thumbnails).zipWithIndex.foreach(println)
  }

  def resizeAndSave(size:(Int,Int))(src:File,dst:File): Unit ={
    if(!dst.exists) {
      try {
        Thumbnails
          .of(src)
          .size(size._1,size._2)
          .keepAspectRatio(false)
          .scalingMode(ScalingMode.BICUBIC)
          .toFile(dst)
      }catch{
        case e:Throwable => e.printStackTrace()
      }
      println("saving resized image:" + dst)
    }else{
      println("skip:"+dst)
    }
  }

  def resizeThumbnails(w:Int,h:Int)(src:File,dst:File): Unit ={
//    val thumbnails = new File("thumbnails").listFiles()
//    val folder = new File("resized")
    val thumbnails = src.listFiles()
    val folder = dst
    folder.mkdirs()
    thumbnails.iterator.foreach{
      case file =>
        val dst = new File(folder.getPath+"/"+file.getName)
        if(!dst.exists) {
          try {
            Thumbnails
              .of(file)
              .size(w,h)
              .keepAspectRatio(false)
              .scalingMode(ScalingMode.BICUBIC)
              .toFile(dst)
          }catch{
            case e:Throwable => e.printStackTrace()
          }
          println("saving resized image:" + dst)
        }else{
          println("skip:"+dst)
        }
    }
  }
}