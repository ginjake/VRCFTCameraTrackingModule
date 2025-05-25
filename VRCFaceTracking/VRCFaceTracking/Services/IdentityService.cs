﻿using Windows.Security.Cryptography;
using Windows.Security.Cryptography.Core;
using Windows.Storage.Streams;
using Windows.System.Profile;
using VRCFaceTracking.Core.Contracts.Services;

namespace VRCFaceTracking.Services;

public class IdentityService : IIdentityService
{
    private string _uniqueUserId = string.Empty;
    
    public string GetUniqueUserId()
    {
        if (!string.IsNullOrEmpty(_uniqueUserId))
        {
            return _uniqueUserId;
        }

        SystemIdentificationInfo systemId = SystemIdentification.GetSystemIdForPublisher();

        // Convert the binary ID to a string
        IBuffer binaryId = systemId.Id;
        string systemIdString = CryptographicBuffer.EncodeToHexString(binaryId);

        // Hash the string
        var hasher = HashAlgorithmProvider.OpenAlgorithm(HashAlgorithmNames.Sha256);
        IBuffer hashed =
            hasher.HashData(CryptographicBuffer.ConvertStringToBinary(systemIdString, BinaryStringEncoding.Utf8));
        _uniqueUserId = CryptographicBuffer.EncodeToHexString(hashed);

        return _uniqueUserId;
    }
}