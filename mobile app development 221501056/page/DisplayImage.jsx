import React, { useState, useRef, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, ActivityIndicator } from 'react-native';
import { Camera, CameraView } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as FileSystem from 'expo-file-system';
import { decodeJpeg } from '@tensorflow/tfjs-react-native';

export default function ObjectDetection() {
    const [hasPermission, setHasPermission] = useState(null);
    const [image, setImage] = useState(null);
    const [detections, setDetections] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const cameraRef = useRef(null);
    const [isTfReady, setIsTfReady] = useState(false);
    const [model, setModel] = useState(null);

    useEffect(() => {
        let isMounted = true;
        (async () => {
            const { status } = await Camera.requestCameraPermissionsAsync();
            if (isMounted) setHasPermission(status === 'granted');

            // Load TensorFlow
            await tf.ready();
            if (isMounted) setIsTfReady(true);

            // Load COCO-SSD model
            const loadedModel = await cocoSsd.load();
            if (isMounted) setModel(loadedModel);
        })();

        return () => {
            // Cleanup function
            isMounted = false;
            setModel(null);
        };
    }, []);

    const takePicture = async () => {
        if (cameraRef.current) {
            const photo = await cameraRef.current.takePictureAsync();
            setImage(photo.uri);
        }
    };

    const detectObjects = async () => {
        if (image && model) {
            setIsLoading(true);
            try {
                const imgB64 = await FileSystem.readAsStringAsync(image, {
                    encoding: FileSystem.EncodingType.Base64,
                });
                const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
                const rawImageData = new Uint8Array(imgBuffer);
                const imageTensor = decodeJpeg(rawImageData);

                const predictions = await model.detect(imageTensor);
                setDetections(predictions);
            } catch (error) {
                console.error('Detection Error:', error);
            } finally {
                setIsLoading(false);
            }
        }
    };

    if (hasPermission === null) {
        return <View />;
    }
    if (hasPermission === false) {
        return <Text>No access to camera</Text>;
    }

    return (
        <View style={styles.container}>
            {!image && (
                <CameraView style={styles.camera} ref={cameraRef}>
                    <View style={styles.buttonContainer}>
                        <TouchableOpacity style={styles.button} onPress={takePicture}>
                            <Text style={styles.text}>Take Photo</Text>
                        </TouchableOpacity>
                    </View>
                </CameraView>
            )}
            {image && (
                <View style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Image source={{ uri: image }} style={styles.previewImage} />
                    <View style={styles.actionButtons}>
                        <TouchableOpacity style={styles.actionButton} onPress={detectObjects}>
                            <Text style={styles.text}>Detect Objects</Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.actionButton} onPress={() => setImage(null)}>
                            <Text style={styles.text}>Capture Another Photo</Text>
                        </TouchableOpacity>
                    </View>
                    {isLoading && <ActivityIndicator size="large" color="#0000ff" />}
                    {detections.length > 0 ? (
                        <View style={styles.detectionResults}>
                            <Text style={styles.detectionText}>Detected Objects:</Text>
                            {detections.map((detection, index) => (
                                <Text key={index} style={styles.detectionItem}>
                                    {detection.class} - {Math.round(detection.score * 100)}%
                                </Text>
                            ))}
                        </View>
                    ) : (
                        !isLoading && (
                            <Text style={styles.noObjectText}>
                                No objects detected. Please try another image.
                            </Text>
                        )
                    )}
                </View>
            )}
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#fff',
    },
    camera: {
        flex: 1,
        width: '100%',
    },
    buttonContainer: {
        flex: 1,
        backgroundColor: 'transparent',
        flexDirection: 'row',
        margin: 20,
    },
    button: {
        flex: 0.3,
        alignSelf: 'flex-end',
        alignItems: 'center',
        backgroundColor: '#f08',
        padding: 10,
        borderRadius: 5,
    },
    text: {
        fontSize: 18,
        color: 'white',
    },
    previewImage: {
        width: 300,
        height: 300,
        borderRadius: 10,
        marginBottom: 20,
    },
    actionButtons: {
        flexDirection: 'row',
        justifyContent: 'space-around',
    },
    actionButton: {
        backgroundColor: '#2196F3',
        padding: 10,
        margin: 10,
        borderRadius: 5,
    },
    detectionResults: {
        marginTop: 20,
        padding: 10,
        backgroundColor: '#f0f0f0',
        borderRadius: 10,
    },
    detectionText: {
        fontSize: 18,
        fontWeight: 'bold',
    },
    detectionItem: {
        fontSize: 16,
        marginVertical: 2,
    },
    noObjectText: {
        marginTop: 20,
        fontSize: 16,
        color: 'red',
    },
});