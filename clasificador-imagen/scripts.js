let net


const imgE1 = document.getElementById('img')
const descE1 = document.getElementById('descripcion_imagen')
const webcamElement = document.getElementById('webcam')
const classifier = knnClassifier.create()

async function app() {
    net = await mobilenet.load()

    var result = await net.classify(imgE1)
    console.log(result)
    displayImagePredicction()

    webcam = await tf.data.webcam(webcamElement)

    while (true) {
        const img = await webcam.capture()
        const result = await net.classify(img)

        const activation = net.infer(img, "conv_preds")
        var result2
        try {
            result2 = await classifier.predictClass(activation)
            const classes = ['Untrained', 'Gato', 'Dino', 'OK', 'Rock']
            document.getElementById('console2').innerHTML = "Console 2 prediccion: " + classes[result2.label]
        } catch (error) {
            console.log('Modelo no configurado aún')
        }

        document.getElementById('console').innerHTML = 'prediccion: ' + result[0].className + ' probability: ' + result[0].probability
    }
    img.dispose()

    await tf.nextFrame()
}


imgE1.onload = async function() {
    displayImagePredicction()
}


async function addExample(classId) {
    console.log('added example')
    const img = await webcam.capture()
    const activation = net.infer(img, true)
    classifier.addExample(activation, classId)

    img.dispose()
}


async function displayImagePredicction() {
    try {
        result = await net.classify(imgE1)
        descE1.innerHTML = JSON.stringify(result)
    } catch (error) {

    }
}


count = 0
async function cambiarImagen(){
    count = count + 1
    imgE1.src = "https://picsum.photos/200/300?random=" + count
}

app()