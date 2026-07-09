using System;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;
using System.Collections.Generic;
using NativeWebSocket;
using System.Text;

public class SAM3Client : MonoBehaviour
{
    [Header("Server")]
    public string serverIP = "172.20.10.2";
    public int serverPort = 8765;

    [Header("Display")]
    public RawImage displayImage;

    [Header("Selection")]
    public Text statusText;

    private WebSocket webSocket;
    private Texture2D frameTexture;
    private byte[] latestFrame;
    private bool hasNewFrame = false;

    private string[] availableObjects = { "door", "hand", "guitar" };
    private int selectedIndex = 0;

    private bool triggerReady = true;
    private bool primaryButtonReady = true;
    private bool secondaryButtonReady = true;

    async void Start()
    {
        Debug.Log("[SAM3] ===== SAM3Client START =====");
        Debug.Log("[SAM3] Connecting to: " + serverIP + ":" + serverPort);

        frameTexture = new Texture2D(640, 480, TextureFormat.RGB24, false);

        webSocket = new WebSocket($"ws://{serverIP}:{serverPort}");

        webSocket.OnMessage += (bytes) =>
        {
            latestFrame = bytes;
            hasNewFrame = true;
        };

        webSocket.OnOpen += () => Debug.Log("[SAM3] Connected to server");
        webSocket.OnError += (e) => Debug.Log($"[SAM3] Error: {e}");
        webSocket.OnClose += (e) => Debug.Log("[SAM3] Disconnected");

        await webSocket.Connect();
        UpdateStatus();
    }

    void Update()
    {
        #if !UNITY_WEBGL || UNITY_EDITOR
            webSocket?.DispatchMessageQueue();
        #endif

        if (hasNewFrame && latestFrame != null)
        {
            frameTexture.LoadImage(latestFrame);
            if (displayImage != null)
                displayImage.texture = frameTexture;
            hasNewFrame = false;
        }

        var rightHand = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
        if (!rightHand.isValid) return;

        // Right index trigger to select
        bool triggerPressed = false;
        rightHand.TryGetFeatureValue(CommonUsages.triggerButton, out triggerPressed);
        if (triggerPressed && triggerReady)
        {
            SelectObject(availableObjects[selectedIndex]);
            triggerReady = false;
        }
        else if (!triggerPressed)
        {
            triggerReady = true;
        }

        // A button - cycle forward
        bool primaryPressed = false;
        rightHand.TryGetFeatureValue(CommonUsages.primaryButton, out primaryPressed);
        if (primaryPressed && primaryButtonReady)
        {
            selectedIndex = (selectedIndex + 1) % availableObjects.Length;
            UpdateStatus();
            primaryButtonReady = false;
        }
        else if (!primaryPressed)
        {
            primaryButtonReady = true;
        }

        // B button - cycle backward
        bool secondaryPressed = false;
        rightHand.TryGetFeatureValue(CommonUsages.secondaryButton, out secondaryPressed);
        if (secondaryPressed && secondaryButtonReady)
        {
            selectedIndex = (selectedIndex - 1 + availableObjects.Length) % availableObjects.Length;
            UpdateStatus();
            secondaryButtonReady = false;
        }
        else if (!secondaryPressed)
        {
            secondaryButtonReady = true;
        }
    }

    void UpdateStatus()
    {
        if (statusText != null)
            statusText.text = "Target: " + availableObjects[selectedIndex] + "\nTrigger to select";
        Debug.Log("[SAM3] Target changed to: " + availableObjects[selectedIndex]);
    }

    void SelectObject(string objectName)
    {
        if (webSocket != null && webSocket.State == WebSocketState.Open)
        {
            string msg = "{\"select\": \"" + objectName + "\"}";
            webSocket.SendText(msg);
            Debug.Log("[SAM3] Selected: " + objectName);

            if (statusText != null)
                statusText.text = "EXECUTING: " + objectName;
        }
    }

    async void OnApplicationQuit()
    {
        if (webSocket != null)
            await webSocket.Close();
    }
}
